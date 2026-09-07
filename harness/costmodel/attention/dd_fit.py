"""DD (box half): the prefill sweep's geometry, and the re-fit it makes possible.

`b9/prefill.py` fitted `fresh ~ 326 M + 1.37 M*S + 72.1*S^2` from S=512 and
S=1024, and eval-plan §2 showed the fit is unidentifiable: two points, two
unknowns, and `bands = p S^2` and `bands = gamma S^2 + delta S` both hit both
points exactly while disagreeing 11% on the coefficient E11's headline rests on.
The S=2048 point that would separate them died at 127.60 GiB RSS on a box with a
128 GiB cap.  This script does two things the two survivors could not support:

  `--geom`   which S tile the band grids exactly, from a Python mirror of
             `prover/src/layout.rs` validated against every artifact's own
             `family_cells.band_leads`.  eval-plan §5.1 asserts S=3072 tiles
             exactly; it does not, and the mirror says why.

  (default)  the re-fit over every measured point: the two competing `bands`
             models and the parameter-free structural one, the linearity of the
             non-band remainder (never testable before -- it WAS the two-point
             fit), the crossover, and E11's multiple under each.

    uv run experiments/b9/dd_fit.py [--geom]

The sweep that feeds it is `b9/gpu/dd_prefill.sh`, run on the H100.
"""
import json
import math
import pathlib
import sys

from harness import inputs

HERE = pathlib.Path(__file__).parent
# `--artifacts DIR`: the sweep outputs at its root, the decode sequence's per-chunk
# JSONs under `seq5/`; the fit is written to `dd_fit.json` beside them.
OUT = inputs.artifacts_dir()

NH, HD, L8B = 32, 128, 32
BANDS = (16, 8, 8)                # nh = 32 as three power-of-two instance bands
GRID_TARGET, CHAN_TARGET = 64, 128
MIN_LOG_ROWS, MAX_LOG_ROWS = 4, 26

# ---------------------------------------------------------------- geometry ----
# A line-for-line mirror of `prover/src/layout.rs`: `Grid::new`, `pick`, and the
# three layouts.  Nothing here is fitted or guessed -- if it disagrees with an
# artifact's `band_leads` the mirror is wrong and `--geom` says so.


def grid(groups, log_rows, cap):
    """`Grid::new(groups, log_rows, cap, _)`."""
    rows = 1 << log_rows
    if rows < groups:
        return None
    chunks = min(rows // groups, max(cap, 1))
    return dict(log_rows=log_rows, rows=rows, groups=groups, chunks=chunks,
                lead=rows - groups * chunks)


def pick(make, key, target):
    """`layout.rs::pick`: the layout whose `key` is closest to `target`, ties to
    the smaller row count.  Ascending `log_rows`, so a tie keeps the first."""
    best = None
    for lr in range(MIN_LOG_ROWS, MAX_LOG_ROWS + 1):
        l = make(lr)
        if l is None:
            continue
        if best is None or abs(key(l) - target) < abs(key(best) - target):
            best = l
    return best


def lay_chan(s, c, hd, insts, target=CHAN_TARGET):
    """`LayoutChan`: one row group per (instance, channel), `pk` keys per row."""
    def make(lr):
        g = grid(hd * insts, lr, min(c, s))
        if g is None:
            return None
        return {**g, "pk": -(-c // g["chunks"]), "pq": -(-s // g["chunks"])}
    return pick(make, lambda l: l["pk"], target)


def lay_key(s, c, hd, insts, target=GRID_TARGET):
    """`LayoutKey`: one row group per (instance, key), `q` scores per row.  This
    is the (t, c) grid -- the region the S^2 term lives in."""
    def make(lr):
        g = grid(c * insts, lr, min(s, hd))
        if g is None:
            return None
        return {**g, "q": -(-s // g["chunks"]), "pv": -(-hd // g["chunks"])}
    return pick(make, lambda l: l["q"], target)


def lay_out(s, c, hd, insts, target=GRID_TARGET):
    """`LayoutOut`: one row group per (instance, output channel)."""
    def make(lr):
        g = grid(hd * insts, lr, s)
        if g is None:
            return None
        return {**g, "q": -(-s // g["chunks"])}
    return pick(make, lambda l: l["q"], min(target, s))


def row_region(s, insts, rw_cfg=16):
    """The per-token region, `attn_trace.rs:155-165`: `rw` tokens per row over
    `insts * S` of them, then padded up to a power of two.  It is the third
    place a shape can fail to tile, and the one `band_leads` does not report."""
    floor = 1 << MIN_LOG_ROWS
    toks = s * insts
    rw = max(min(rw_cfg, toks // floor), 1)
    rows = max(-(-toks // rw), floor)
    assert rw * rows == toks, f"per-token region does not tile: rw={rw} rows={rows}"
    log_row = max((rows - 1).bit_length(), MIN_LOG_ROWS)
    return dict(rw=rw, rows=rows, log_row=log_row, rrows=1 << log_row,
                lead=(1 << log_row) - rows)


def band_base_geom(s, c, hd=HD, bands=BANDS, rows_region=True, rw_cfg=16):
    """The band family's committed BASE cells read off the realised layouts --
    the same three region sums the closed form below collapses to when every
    grid tiles exactly, but valid when they do not.  `attn_trace.rs:1170-1174`:
    the channel region's `(pk + pq)`, the key region's `(2q + pv)`, and the
    per-token region's `4 * rw`; the `out` region is billed to `attnout`."""
    tot = 0
    for i in bands:
        lc, lk = lay_chan(s, c, hd, i), lay_key(s, c, hd, i)
        tot += (lc["pk"] + lc["pq"]) * lc["rows"] + (2 * lk["q"] + lk["pv"]) * lk["rows"]
        if rows_region:
            r = row_region(s, i, rw_cfg)
            tot += 4 * r["rw"] * r["rrows"]
    return tot


def geometry(s, c, hd=HD, bands=BANDS):
    """The three regions of every band, as the composition lays them out."""
    return [dict(insts=i,
                 chan=lay_chan(s, c, hd, i),
                 key=lay_key(s, c, hd, i),
                 out=lay_out(s, c, hd, i)) for i in bands]


def leads(s, c, hd=HD, bands=BANDS):
    return [(b["chan"]["lead"], b["key"]["lead"], b["out"]["lead"])
            for b in geometry(s, c, hd, bands)]


def tiles_exactly(s, c=None, hd=HD, bands=BANDS):
    return all(l == (0, 0, 0) for l in leads(s, c if c else s, hd, bands))


def composable(s, c=None, hd=HD, bands=BANDS, rw_cfg=16):
    """Will the AIR build this shape at all?  `attn_trace.rs:320-326` asserts the
    grid's and the output region's TOKEN axes tile exactly --

        assert_eq!(l.key.q * l.key.g.chunks, sh.s,
                   "grid layout leaves padded token slots")

    -- because a padded token slot would have to commit a numerator that is both
    a table output and zero, and the table's zero row is not the one a zero shift
    maps to.  `attn_trace.rs:159` asserts the per-token region tiles too.  This is
    a stronger condition than `Grid::lead == 0`, which is about the row-group
    (key) axis; a shape can fail this and still report `lead 0` everywhere."""
    c = c if c else s
    for i in bands:
        lk, lo = lay_key(s, c, hd, i), lay_out(s, c, hd, i)
        if lk["q"] * lk["chunks"] != s or lo["q"] * lo["chunks"] != s:
            return False
        try:
            row_region(s, i, rw_cfg)
        except AssertionError:
            return False
    return True


# -------------------------------------------------------- the band closed form -
# `prefill.py`'s structural model, verified exact against 8 artifacts across two
# harnesses and three contexts.  Zero free parameters; every coefficient is read
# off `attn_trace.rs:1167-1181` with `layout.rs`.  Valid only where the regions
# tile exactly -- which is the point of `--geom`.
NH_HD = NH * HD                                    # 4,096


def band_base(s, c, rows_region=True):
    return (2 * NH * s * c + 2 * NH_HD * c + NH_HD * s
            + (4 * NH * s if rows_region else 0))


def band_inter(s, c, rows_region=True):
    return (76 * s * c + (2 * NH_HD + 384 + 8192) * c + 2 * NH_HD * s
            + (272 * s if rows_region else 0))


def band_fresh(s, c, rows_region=True):
    return band_base(s, c, rows_region) + band_inter(s, c, rows_region)


GAMMA = 2 * NH                                     # 64  = the S*C base coefficient
DELTA = 2 * NH_HD + NH_HD + 4 * NH                 # 12,416 at C = S
BAND_SC, BAND_C, BAND_S = 140, 24960, 12688        # band FRESH: base + interaction


# ------------------------------------------------------------------ loading ---
# Three families of artifact, kept apart on purpose:
#   prefill_s*.json  the 2026-08-22 sweep, B9's `b65_ref` tree (the survivors)
#   dd_s*.json       this sweep, `prover/` after B10 and B11 A-Z
#   dd_p*.json       the same, `ao_rem_bits` pinned so the family is parameter
#                    identical -- the one shape parameter that moved originally
SETS = (("b9", "prefill_s{}.json"), ("dd", "dd_s{}.json"), ("ddpin", "dd_p{}.json"))
SS = (256, 512, 1024, 1536, 2048, 3072, 4096)


def load(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        d = json.loads(path.read_text())["result"]["point"]
    except Exception:
        return None
    fc = d["family_cells"]
    p = d["params"]
    return dict(
        s=p["s"], c=p["c"], accepts=d["accepts"], off_table=d["off_table"],
        fresh=d["cells"]["fresh"], base=d["cells"]["base"],
        inter=d["cells"]["interaction"], pre=d["cells"]["preprocessed"],
        bands=fc["bands"], dequant=fc["dequant"], freivalds=fc["freivalds"],
        attnout=fc["attnout"],
        band_inter=(fc.get("interaction_by_family", {}).get("bands", {}) or {}).get("cells"),
        band_leads=[(l["chan"], l["key"], l["out"]) for l in fc["band_leads"]],
        ao_rem_bits=p["ao_rem_bits"], kmax=p["kmax"], log_table=p["log_table"],
        peak_rss=d.get("peak_rss_bytes"), wall=d["phases"]["wall_secs"],
        prove=d["phases"].get("prove_call_secs"),
        sha=d.get("proof_sha256"), path=str(path))


def load_all():
    got = {}
    for tag, pat in SETS:
        got[tag] = {}
        for s in SS:
            r = load(OUT / pat.format(s))
            if r:
                got[tag][s] = r
    return got


def g(x, unit=""):
    for sfx, d in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.3f} {sfx}{unit}"
    return f"{x:.2f}{unit}"


# ------------------------------------------------------------------- --geom ---
def geom_report():
    print("=" * 92)
    print("DD: which S tile the band grids exactly (mirror of prover/src/layout.rs)")
    print("=" * 92)
    print("C = S, hd = 128, bands 16/8/8, scores-per-row 64, keys-per-row 128.")
    print("The (t, c) grid is `LayoutKey`: one row group per (instance, key), so its")
    print("group count is `C * insts` -- and a power-of-two row count can only tile")
    print("that exactly when C has no odd factor.  `chunks` is capped at min(S, hd) =")
    print("128, which is why the padding does not shrink away at larger S.")
    print()
    print(f"  {'S = C':>6} {'tiles':>7} {'AIR':>7} {'band':>5} {'key log_rows':>13} "
          f"{'chunks':>7} {'scores/row':>11} {'key lead':>10} {'lead frac':>10} "
          f"{'chan/out lead':>14} {'per-tok lead':>13}")
    rows = {}
    for s in SS:
        ex, cmp_ = tiles_exactly(s), composable(s)
        rows[s] = dict(exact=ex, composable=cmp_, bands=[],
                       bands_base_layout=band_base_geom(s, s),
                       bands_base_closed=band_base(s, s))
        for b in geometry(s, s):
            k, ch, ou = b["key"], b["chan"], b["out"]
            rr = row_region(s, b["insts"])
            rows[s]["bands"].append(dict(
                insts=b["insts"], key_log_rows=k["log_rows"], key_chunks=k["chunks"],
                scores_per_row=k["q"], key_lead=k["lead"],
                key_lead_frac=k["lead"] / k["rows"], chan_lead=ch["lead"],
                out_lead=ou["lead"], row_lead=rr["lead"], row_rows=rr["rrows"]))
            print(f"  {s if b['insts'] == BANDS[0] else '':>6} "
                  f"{('EXACT' if ex else 'padded') if b['insts'] == BANDS[0] else '':>7} "
                  f"{(('builds' if cmp_ else 'REFUSES') if b['insts'] == BANDS[0] else ''):>7} "
                  f"{b['insts']:>5} {k['log_rows']:>13} {k['chunks']:>7} {k['q']:>11} "
                  f"{k['lead']:>10} {k['lead'] / k['rows'] * 100:>9.4f}% "
                  f"{str((ch['lead'], ou['lead'])):>14} "
                  f"{str(rr['lead']) + '/' + str(rr['rrows']):>13}")
        gm, cl = rows[s]["bands_base_layout"], rows[s]["bands_base_closed"]
        print(f"  {'':>6} {'':>7} {'band base from the layouts':>31}: {gm:>15,}"
              f"   exact-tiling closed form {cl:>15,}"
              f"   {'equal' if gm == cl else f'+{gm - cl:,} = +{(gm / cl - 1) * 100:.3f}%'}")
    print()
    print("  So the exactly-tiling shapes reachable at all are the powers of two.  Every")
    print("  S strictly between two of them carries an odd factor into `C * insts`, and")
    print("  eval-plan §5.1's `jump to S=3072, which tiles exactly` is wrong for exactly")
    print("  the reason it gives for 1536: 3072 = 3*2^10, and 3 never divides 2^log_rows.")
    print()
    print("  And 1536 is not merely biased -- the AIR REFUSES it.  `attn_trace.rs:320`")
    print("  asserts the grid's TOKEN axis tiles (`q * chunks == S`; 74*21 = 1554 at")
    print("  S=1536), because a padded token slot would have to commit a numerator that")
    print("  is both a table output and zero.  So §5.1's first option -- accept the")
    print("  padding and arm the controls -- is not an option that exists.  Measured:")
    print("  the S=1536 point in this sweep dies at that assertion in 20 s.")
    print()
    print("  the two conditions are different, and rule 13's precondition needs the")
    print("  SECOND without the first -- a shape the AIR builds that still leaves an")
    print("  inert lead row.  Scanning C = S in steps of 16 up to 8192:")
    build = [s for s in range(16, 8193, 16) if composable(s)]
    lead13 = [s for s in build if not tiles_exactly(s)]
    print(f"    {len(build)} of 512 shapes build; {len(lead13)} of those leave a lead row.")
    frac = lambda s: max(b["key"]["lead"] / b["key"]["rows"] for b in geometry(s, s)) * 100
    print("    a few, with the largest band lead fraction: "
          + ", ".join(f"{s} ({frac(s):.2f}%)" for s in (240, 336, 480, 1600, 2016)
                      if s in lead13))
    print("    so rule 13 IS reachable -- just not at 1536 or 3072, and never at a")
    print("    power of two.  This sweep arms the controls at S = C = 240.")
    print()
    print("  the mirror against every artifact that records `band_leads`, including")
    print("  B10's seq5 chunks -- the real 8B witness at C != S, three prefix bands plus")
    print("  three corner bands at C = S, which is the only check of the C != S arm:")
    ok = bad = 0
    checks = [(tag, s, r) for tag, _ in SETS for s, r in sorted(load_all()[tag].items())]
    seq5 = OUT / "seq5"
    for lay in (0, 16, 31):
        for c in (4096, 16384):
            r = load(seq5 / f"seq_c{c}_L{lay}.json")
            if r:
                checks.append((f"seq5 L{lay}", r["s"], r))
    for tag, s, r in checks:
        pred = leads(r["s"], r["c"])
        if len(r["band_leads"]) == 2 * len(pred):        # a corner rides at C = S
            pred = pred + leads(r["s"], r["s"])
        good = pred == r["band_leads"]
        ok, bad = ok + good, bad + (not good)
        print(f"    {tag:9s} S={r['s']:<5d} C={r['c']:<6d} "
              f"{'ok' if good else 'MISMATCH: ' + str(pred) + ' vs ' + str(r['band_leads'])}"
              f"   ({len(pred)} bands, all {'0/0/0' if not any(any(t) for t in pred) else 'padded'})")
    print(f"    {ok} agree, {bad} disagree.")
    return rows


# ------------------------------------------------------------------ the fit ---
def lstsq(cols, y):
    """Least squares by normal equations; `cols` is a list of basis vectors."""
    n = len(cols)
    a = [[sum(cols[i][k] * cols[j][k] for k in range(len(y))) for j in range(n)]
         for i in range(n)]
    b = [sum(cols[i][k] * y[k] for k in range(len(y))) for i in range(n)]
    for i in range(n):                                   # Gaussian elimination
        piv = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[piv] = a[piv], a[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, n):
            f = a[r][i] / a[i][i]
            for cc in range(i, n):
                a[r][cc] -= f * a[i][cc]
            b[r] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def cross_of(a2, a1, a0):
    """The S where `a2 S^2 + a1 S + a0 = 0`, i.e. where the band family passes
    half the chunk."""
    return (-a1 + math.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2)


def report():
    got = load_all()
    print("=" * 92)
    print("DD: the prefill sweep re-fit -- three or more points instead of two")
    print("=" * 92)
    for tag, _ in SETS:
        if not got[tag]:
            continue
        print(f"\n{tag}:")
        print(f"  {'S = C':>6} {'tiles':>6} {'fresh':>11} {'bands(base)':>12} "
              f"{'pre':>10} {'peak RSS':>10} {'wall':>8} {'ao_rem':>7} {'acc':>5} "
              f"{'lead rows':>11}")
        for s, r in sorted(got[tag].items()):
            print(f"  {s:>6} {('exact' if tiles_exactly(s) else 'padded'):>6} "
                  f"{g(r['fresh']):>11} {g(r['bands']):>12} {g(r['pre']):>10} "
                  f"{(r['peak_rss'] / 2**30 if r['peak_rss'] else 0):>9.2f}G "
                  f"{r['wall']:>7.1f}s {r['ao_rem_bits']:>7} {str(r['accepts']):>5} "
                  f"{str(sorted(set(r['band_leads']))):>11}")

    # -- the reproduction control: does today's tree give the 2026-08-22 numbers?
    print()
    print("the reproduction control -- this tree against the archived survivors")
    print("  (the archived points are B9's `b65_ref` at 2026-08-22; this sweep is")
    print("  `prover/` after B10 and B11 A-Z.  If they differ the third point is not")
    print("  in their family and the fit is done on this sweep alone.)")
    repro = []
    for s in sorted(set(got["b9"]) & set(got["dd"])):
        a, b = got["b9"][s], got["dd"][s]
        same = {k: a[k] == b[k] for k in ("fresh", "bands", "pre", "base", "inter")}
        repro.append(dict(s=s, **{f"{k}_same": v for k, v in same.items()},
                          b9_fresh=a["fresh"], dd_fresh=b["fresh"],
                          b9_bands=a["bands"], dd_bands=b["bands"],
                          b9_sha=a["sha"], dd_sha=b["sha"], sha_same=a["sha"] == b["sha"]))
        print(f"  S={s:<5d} fresh {a['fresh']:>15,} vs {b['fresh']:>15,} "
              f"{'SAME' if same['fresh'] else 'MOVED'};  bands {a['bands']:>13,} vs "
              f"{b['bands']:>13,} {'SAME' if same['bands'] else 'MOVED'};  "
              f"digest {'SAME' if a['sha'] == b['sha'] else 'MOVED'}")
    if not repro:
        print("  (no overlap yet)")

    # -- the one confound §5.1 named: `ao_rem_bits` is data-derived and moved
    #    17 -> 18 -> 20 across the family.  Pinning it at one value above every
    #    point's own makes the family parameter identical.  It widens the
    #    attnout region and the `o` Freivalds table, so `fresh` must move and `bands`
    #    must NOT -- which is the check.
    pinned, pinned_fit = [], None
    if got["ddpin"]:
        print()
        print("the `ao_rem_bits` confound, removed -- the same points at one pinned value")
        print(f"  {'S':>6} {'ao_rem nat/pin':>15} {'bands nat':>14} {'bands pin':>14} "
              f"{'fresh nat':>15} {'fresh pin':>15} {'d fresh':>9}")
        for s in sorted(set(got["dd"]) & set(got["ddpin"])):
            a, b = got["dd"][s], got["ddpin"][s]
            pinned.append(dict(s=s, bands_same=a["bands"] == b["bands"],
                               nat_ao=a["ao_rem_bits"], pin_ao=b["ao_rem_bits"],
                               nat_fresh=a["fresh"], pin_fresh=b["fresh"],
                               nat_bands=a["bands"], pin_bands=b["bands"]))
            ao = f"{a['ao_rem_bits']}/{b['ao_rem_bits']}"
            print(f"  {s:>6} {ao:>15} "
                  f"{a['bands']:>14,} {b['bands']:>14,}"
                  f"{'' if a['bands'] == b['bands'] else ' MOVED'} "
                  f"{a['fresh']:>15,} {b['fresh']:>15,} "
                  f"{b['fresh'] / a['fresh'] - 1:>+8.3%}")
        print("  `bands` unmoved at every point is the confound shown to be outside the")
        print("  band family; what moves is the attention-output region alone.")
        pin_ex = {s: got["ddpin"][s] for s in sorted(got["ddpin"])
                  if tiles_exactly(s) and got["ddpin"][s]["accepts"]}
        if len(pin_ex) >= 3:
            pss = sorted(pin_ex)
            pr = {s: pin_ex[s]["fresh"] - band_fresh(s, s) for s in pss}
            pq, pf = lstsq([[float(s) for s in pss], [1.0] * len(pss)],
                           [pr[s] for s in pss])
            px = cross_of(BAND_SC, BAND_S + BAND_C - pq, -pf)
            print(f"  the whole fit redone on the pinned family: rest = {pf:,.0f} + "
                  f"{pq:,.0f} S, crossover S ~ {px:,.0f}")
            pinned_fit = dict(fixed=pf, per_token=pq, crossover=px, points=pss)

    # -- which set to fit on
    src = "dd" if len(got["dd"]) >= 3 else "b9"
    pts = dict(got["b9"])
    pts.update(got[src])                       # the new sweep wins where both exist
    exact = {s: r for s, r in sorted(pts.items()) if tiles_exactly(s) and r["accepts"]}
    padded = {s: r for s, r in sorted(pts.items()) if not tiles_exactly(s)}
    if len(exact) < 3:
        print("\n  fewer than three exactly-tiling points; the fit stays unidentifiable.")
        return None
    print(f"\nfitting on {len(exact)} exactly-tiling points: {sorted(exact)}"
          f"{f' (padded, held out: {sorted(padded)})' if padded else ''}")

    ss = sorted(exact)
    y = [exact[s]["bands"] for s in ss]
    # M1, the incumbent: bands = p S^2, one free parameter
    (p1,) = lstsq([[s * s for s in ss]], y)
    # M2: bands = gamma S^2 + delta S, two free parameters
    g2, d2 = lstsq([[s * s for s in ss], [float(s) for s in ss]], y)
    # M3, structural: zero free parameters
    print()
    print("model selection on the band family (BASE cells), C = S")
    print(f"  {'S':>6} {'measured':>15} {'M1 p S^2':>15} {'resid':>10} "
          f"{'M2 gS^2+dS':>15} {'resid':>8} {'M3 structural':>15} {'resid':>8}")
    res = dict(m1=[], m2=[], m3=[])
    for s in ss:
        m1, m2, m3 = p1 * s * s, g2 * s * s + d2 * s, band_base(s, s)
        for k, v in (("m1", m1), ("m2", m2), ("m3", m3)):
            res[k].append(exact[s]["bands"] - v)
        print(f"  {s:>6} {exact[s]['bands']:>15,} {m1:>15,.0f} "
              f"{(m1 / exact[s]['bands'] - 1) * 100:>9.3f}% {m2:>15,.0f} "
              f"{(m2 / exact[s]['bands'] - 1) * 100:>7.3f}% {m3:>15,} "
              f"{(m3 / exact[s]['bands'] - 1) * 100:>7.3f}%")

    def rms(v):
        return math.sqrt(sum(x * x for x in v) / len(v))
    print(f"  free parameters      {'1':>15} {'':>10} {'2':>15} {'':>8} {'0':>15}")
    print(f"  RMS residual (cells) {rms(res['m1']):>15,.0f} {'':>10} "
          f"{rms(res['m2']):>15,.0f} {'':>8} {rms(res['m3']):>15,.0f}")
    print(f"  M1 p = {p1:.4f} against M2 gamma = {g2:.4f} and the structural "
          f"{GAMMA} ({p1 / GAMMA - 1:+.1%})")
    print(f"  M2 delta = {d2:,.0f} against the structural {DELTA:,}")
    print()
    print("  the same family's INTERACTION cells, where the artifact records them.")
    print("  The 2026-08 runs predate `interaction_by_family`, so S=2048 is the first")
    print("  C = S shape at which the interaction closed form can be checked at all:")
    for s in ss:
        gi = exact[s]["band_inter"]
        if gi is None:
            print(f"    {s:>5}  (not recorded in this artifact)")
            continue
        pi = band_inter(s, s)
        print(f"    {s:>5}  predicted {pi:>15,}   measured {gi:>15,}"
              f"   {'exact' if pi == gi else 'MISMATCH'}")
    print()
    print("  the incumbent `p` is a SECANT, and the sweep watches it move.  Read")
    print("  between each adjacent pair it is `gamma + delta/(S1+S2)`, so it falls")
    print("  towards 64 as the pair moves out -- it was never a coefficient:")
    secants = []
    for i in range(len(ss) - 1):
        a_, b_ = ss[i], ss[i + 1]
        pm = (exact[b_]["bands"] - exact[a_]["bands"]) / (b_ * b_ - a_ * a_)
        secants.append(dict(lo=a_, hi=b_, p=pm, closed=GAMMA + DELTA / (a_ + b_)))
        print(f"    {a_:>5} -> {b_:<5}  p = {pm:>8.4f}   gamma + delta/(S1+S2) = "
              f"{GAMMA + DELTA / (a_ + b_):>8.4f}   {'exact' if abs(pm - GAMMA - DELTA / (a_ + b_)) < 1e-9 else 'DIFFERS'}")
    pub = 72.0833333
    if max(ss) >= 2048:
        pred_inc = pub * 2048 ** 2
        meas = exact[2048]["bands"]
        print(f"\n  the published p = 72.083 predicts bands(2048) = {pred_inc:,.0f};")
        print(f"  the measurement is {meas:,} -- the incumbent is out by "
              f"{pred_inc - meas:,.0f} cells ({pred_inc / meas - 1:+.2%}), and the")
        print(f"  structural model is out by {band_base(2048, 2048) - meas}.")

    # -- the remainder, which is what the crossover actually turns on
    print()
    print("the non-band remainder `rest = fresh - band_fresh(S, C)`, and whether it is")
    print("linear in S.  This was NEVER testable: with two points a linear fit is exact")
    print("by construction, and `prefill.py` says so in its own comment.")
    rest = {s: exact[s]["fresh"] - band_fresh(s, s) for s in ss}
    q, f0 = lstsq([[float(s) for s in ss], [1.0] * len(ss)], [rest[s] for s in ss])
    print(f"  rest(S) = {g(f0)}+ {q:,.0f} S")
    print(f"  {'S':>6} {'rest measured':>16} {'linear fit':>16} {'resid':>10} "
          f"{'of fresh':>10}")
    rest_res = []
    for s in ss:
        m = f0 + q * s
        rest_res.append(rest[s] - m)
        print(f"  {s:>6} {rest[s]:>16,} {m:>16,.0f} {(m / rest[s] - 1) * 100:>9.4f}% "
              f"{(m - rest[s]) / exact[s]['fresh'] * 100:>9.4f}%")
    print(f"  RMS residual {rms(rest_res):,.0f} cells "
          f"({rms(rest_res) / max(exact[s]['fresh'] for s in ss) * 100:.4f}% of the "
          f"largest chunk)")

    # -- the whole-chunk model and what it implies
    print()
    print("the whole chunk, and the crossover -- where the band family passes half of it")

    def fresh_of(s, c):
        return f0 + q * s + band_fresh(s, c)
    for s in ss:
        m = fresh_of(s, s)
        print(f"    S = C = {s:>5}: model {m:>16,.0f}   measured {exact[s]['fresh']:>16,}   "
              f"{m / exact[s]['fresh'] - 1:+.4%}")

    # the incumbent, rebuilt exactly as `prefill.py` does it, from the two survivors
    a, b = got["b9"].get(512), got["b9"].get(1024)
    inc = None
    if a and b:
        p_i = (b["bands"] - a["bands"]) / (b["s"] ** 2 - a["s"] ** 2)
        q_i = ((b["fresh"] - b["bands"]) - (a["fresh"] - a["bands"])) / (b["s"] - a["s"])
        f_i = (a["fresh"] - a["bands"]) - q_i * a["s"]
        x_i = cross_of(p_i, -q_i, -f_i)
        inc = dict(p=p_i, q=q_i, fix=f_i, crossover=x_i)
        print(f"\n  incumbent (2 points, base bands against a fresh rest)  S ~ {x_i:>9,.0f}")
    # The incumbent's question is ill-posed, and four points is what shows it.
    # It sets a BASE band count against `fresh - bands_base`, and that remainder
    # still contains the band family's own INTERACTION, which grows as 76 S^2.
    # So it fits a quadratic with a line: the answer is whatever pair you secant.
    rest_b = {s: exact[s]["fresh"] - band_base(s, s) for s in ss}
    qb, fb = lstsq([[float(s) for s in ss], [1.0] * len(ss)], [rest_b[s] for s in ss])
    pair_x = []
    for i in range(len(ss) - 1):
        a_, b_ = ss[i], ss[i + 1]
        p_ = (exact[b_]["bands"] - exact[a_]["bands"]) / (b_ * b_ - a_ * a_)
        q_ = (rest_b[b_] - rest_b[a_]) / (b_ - a_)
        f_ = rest_b[a_] - q_ * a_
        pair_x.append(dict(lo=a_, hi=b_, p=p_, crossover=cross_of(p_, -q_, -f_)))
        print(f"  the same construction on the {a_:>5} / {b_:<5} pair             "
              f"S ~ {pair_x[-1]['crossover']:>9,.0f}")
    print("  -- it moves by a third across the sweep, because `fresh - bands_base`")
    print("  keeps the band family's own interaction (76 S^2) and is not linear:")
    print(f"     a line through all four misses it by {rms([rest_b[s] - (fb + qb * s) for s in ss]):,.0f} "
          f"cells RMS, against {rms(rest_res):,.0f} for the fresh remainder.")
    print("  Setting a base count against a fresh one has no crossover at all when")
    print("  it is written out: `bands_base = fresh/2` is `12 S^2 + (12,816 + q) S")
    print("  + F = 0`, which has no positive root -- the family's interaction alone")
    print("  outgrows its base.  The quantity was never well posed.")
    x_str = cross_of(GAMMA, DELTA - qb, -fb)
    x_fresh = cross_of(BAND_SC, BAND_S + BAND_C - q, -f0)
    print()
    print(f"  fresh against fresh, all four points (THE one to quote)  "
          f"S ~ {x_fresh:>9,.0f}")
    # How much of that is the data and how much is the fit?  Drop each point in
    # turn and refit the remainder; the spread is the honest precision.
    jack = []
    for drop in ss:
        keep = [s for s in ss if s != drop]
        qj, fj = lstsq([[float(s) for s in keep], [1.0] * len(keep)],
                       [rest[s] for s in keep])
        jack.append((drop, cross_of(BAND_SC, BAND_S + BAND_C - qj, -fj)))
    lo, hi = min(x for _, x in jack), max(x for _, x in jack)
    print(f"  leave-one-out on the remainder: "
          + ", ".join(f"-{d}: {x:,.0f}" for d, x in jack))
    print(f"  spread {lo:,.0f} - {hi:,.0f}, i.e. +/-{(hi - lo) / 2 / x_fresh:.2%} -- "
          f"two significant figures are earned.")

    # -- E11
    e11 = {}
    try:
        from harness.constants import value as frozen
        decode = {4096: frozen("cells.c4096.fresh_33chunks"),
                  16384: frozen("cells.c16384.fresh_33chunks")}
    except Exception as exc:
        decode = {}
        print(f"\n  (E11 multiple skipped -- {type(exc).__name__}: {exc})")
    if decode:
        E11_S = 1024
        print()
        print(f"E11's cells multiple, blocks of S = {E11_S:,}, against seq5's measured")
        print("512-token decode sequence (H200, 33 chunks, fresh cells), 32 layer chunks.")
        print(f"  {'C = N':>6} {'prompt cells':>14} {'x decode':>9} {'prompt+output':>15} "
              f"{'x decode':>9} {'square-chunk read':>18}")
        for n_tok in sorted(decode):
            n = n_tok // E11_S
            prompt = sum(fresh_of(E11_S, (i + 1) * E11_S) for i in range(n)) * L8B
            square = fresh_of(E11_S, E11_S) * n * L8B
            e11[n_tok] = dict(prompt_cells=prompt, decode_cells=decode[n_tok],
                              x_prompt=prompt / decode[n_tok],
                              x_total=(prompt + decode[n_tok]) / decode[n_tok],
                              x_square=square / decode[n_tok])
            print(f"  {n_tok:>6} {g(prompt):>14} {prompt / decode[n_tok]:>8.2f}x "
                  f"{g(prompt + decode[n_tok]):>15} "
                  f"{(prompt + decode[n_tok]) / decode[n_tok]:>8.2f}x "
                  f"{square / decode[n_tok]:>17.2f}x")

    # -- the padded points, held out: what the geometry costs the term being fitted
    if padded:
        print()
        print("the padded shapes, held OUT of the fit -- eval-plan §5.1's bias, measured")
        print("rather than simulated.  The closed form is the exactly-tiling one, so the")
        print("excess IS the inert padding entering `bands`, the term being fitted, and")
        print("`layout` is what the realised layouts predict it to be.")
        print(f"  {'S':>6} {'bands measured':>16} {'layout predicts':>16} "
              f"{'exact-tiling form':>18} {'excess':>12} {'excess %':>9}")
        for s, r in padded.items():
            if r.get("bands") is None:
                continue
            ex, gm = band_base(s, s), band_base_geom(s, s)
            print(f"  {s:>6} {r['bands']:>16,} {gm:>16,}"
                  f"{'' if gm == r['bands'] else ' MISMATCH'} {ex:>18,} "
                  f"{r['bands'] - ex:>12,} {(r['bands'] / ex - 1) * 100:>8.3f}%")
        print("  §5.1 priced the bias at the 1.5625% inert row fraction.  It is larger,")
        print("  because the realised `scores_per_row` also steps (64 -> 74) and the")
        print("  per-token region pads separately -- `band_leads` reports neither.")

    blob = dict(
        sets={t: {str(s): r for s, r in v.items()} for t, v in got.items()},
        repro=repro, pinned=pinned, pinned_fit=pinned_fit,
        fitted_on=sorted(exact), padded_held_out=sorted(padded),
        band_models=dict(
            m1_p=p1, m2_gamma=g2, m2_delta=d2, structural_gamma=GAMMA,
            structural_delta=DELTA,
            rms=dict(m1=rms(res["m1"]), m2=rms(res["m2"]), m3=rms(res["m3"])),
            resid={k: v for k, v in res.items()}),
        rest=dict(fixed=f0, per_token=q, rms=rms(rest_res),
                  measured={str(s): rest[s] for s in ss}),
        crossover=dict(incumbent=inc["crossover"] if inc else None,
                       structural_base=x_str, structural_fresh=x_fresh),
        rest_base=dict(fixed=fb, per_token=qb), secants=secants,
        pair_crossovers=pair_x, crossover_jackknife=jack,
        incumbent=inc, e11=e11,
        geometry={str(s): dict(exact=tiles_exactly(s), leads=leads(s, s)) for s in SS})
    OUT.mkdir(exist_ok=True)
    (OUT / "dd_fit.json").write_text(json.dumps(blob, indent=1))
    print(f"\nwrote {OUT / 'dd_fit.json'}")
    return blob


if __name__ == "__main__":
    if "--geom" in sys.argv:
        geom_report()
    else:
        geom_report()
        print()
        report()
