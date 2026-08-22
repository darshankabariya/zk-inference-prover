"""B10 step 1: the model-wide shape table, read off the 32 manifests.

A chunk loop over a real model may mix table sizes but not two statements
(`b10/gpu/m5/chunks.rs`), so `kmax` and `ao_rem_bits` are pinned at the
model-wide worst case while `master_table_bits` varies per layer.  This reads
what each layer needs -- before a loop is launched, not from the assert it
trips -- and prints the table B10's record carries.

`ao_rem_bits` is `Layer::rem_bits()` in `b65/bench/src/layer.rs`:

    bits( 2 * max_{h,t,j} ( Z[h,t] << max(-(jv[h,j] - jo[t]), 0) ) )

which for a fixed (h, t) is maximised at the smallest `jv[h, .]`, so the whole
maximum is over (h, t) alone.

    uv run python b10/gpu/shapes32.py /root/b8/w8b/c16384/witness_L 32
"""
import json
import os
import sys

import numpy as np


def shape_of(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    nh, s = man["heads"], man["s"]
    zz = np.fromfile(os.path.join(d, "zz.i32"), dtype=np.int32).reshape(nh, s).astype(np.int64)
    jo = np.fromfile(os.path.join(d, "jo.i32"), dtype=np.int32).astype(np.int64)      # (S,)
    jv = np.fromfile(os.path.join(d, "jv.i32"), dtype=np.int32).reshape(nh, -1)        # (nh, hd)
    shift = np.maximum(jo[None, :] - jv.min(-1)[:, None], 0)                           # (nh, S)
    rem = int(2 * (zz << shift).max())
    accs = man["accumulators"]
    return dict(
        layer=man["layer"], c=man["c"], s=man["s"],
        table_bits=man["master_table_bits"], spread=man["frac_spread"],
        d_bits=man["d_bits"], z_max=man["z_max"],
        ao_rem_bits=int(rem).bit_length(),
        kmax_needed=max(v["kmax_needed"] for v in accs.values()),
        kmax_blind=max(v["kmax_worst_case"] for v in accs.values()),
        fold_q_max=man["attn_out"]["fold_q_max"],
        slim=man.get("slim", False),
    )


if __name__ == "__main__":
    base, n = sys.argv[1], int(sys.argv[2])
    rows = [shape_of(f"{base}{k}") for k in range(n)]
    print(f"{'L':>3} {'C':>6} {'S':>4} {'tbl':>4} {'sprd':>5} {'dbits':>6} {'zmax':>8} "
          f"{'aorem':>6} {'kmax':>5} {'blind':>6} {'foldq':>6}")
    for r in rows:
        print(f"{r['layer']:>3} {r['c']:>6} {r['s']:>4} {r['table_bits']:>4} "
              f"{r['spread']:>5.1f} {r['d_bits']:>6} {r['z_max']:>8} {r['ao_rem_bits']:>6} "
              f"{r['kmax_needed']:>5} {r['kmax_blind']:>6} {r['fold_q_max']:>6}")
    tb = sorted({r["table_bits"] for r in rows})
    hist = {b: sum(1 for r in rows if r["table_bits"] == b) for b in tb}
    worst = dict(ao_rem_bits=max(r["ao_rem_bits"] for r in rows),
                 kmax=max(r["kmax_blind"] for r in rows),
                 fold_q_max=max(r["fold_q_max"] for r in rows),
                 table_bits=hist)
    # 5.010 fresh cells per master-table row (B9 1.5), so what the tables cost
    # per-layer against forcing the model's largest on all of them
    per = {b: 5.010 * (1 << b) for b in tb}
    own = sum(per[r["table_bits"]] for r in rows)
    one = per[max(tb)] * len(rows)
    print(f"\nworst case: {json.dumps(worst)}")
    print(f"master table, per-layer {own / 1e9:.3f} G cells vs one shape at 2^{max(tb)} "
          f"{one / 1e9:.3f} G = x{one / own:.2f}")
    assert all(r["fold_q_max"] <= 127 for r in rows), "a layer's fold quotient is past 127"
    out = os.path.join(os.path.dirname(base) or ".", "shapes.json")
    json.dump(dict(rows=rows, worst=worst), open(out, "w"), indent=1)
    print(f"-> {out}")
