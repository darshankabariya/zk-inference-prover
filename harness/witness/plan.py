#!/usr/bin/env python3
"""Stream O: write the chunk loop's plan.

One line per chunk, in the order the sequence proves them: the 32 layers and
then the head. A line carries only what that chunk decides for itself -- its
witness, its master table, and for the head the argmax and its row count.
Everything the whole sequence shares (the round, the band sizes, the pins, the
tight table) stays on the command line, because that is where the loop's
shared arguments live and repeating them per line is how a driver drifts.

The lines are the same arguments `n_seq.sh`'s `seq` step passes to 33 separate
`compose` processes, which is what makes the loop's digests comparable to
theirs chunk by chunk.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="the layer witness directory, holding shapes.json and witness_L*")
    ap.add_argument("--head", default="", help="the head witness directory, if it is to be proved")
    ap.add_argument("--layers", default="", help="which layers, e.g. 0-31 or 0,1,2 (default all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = {r["layer"]: r for r in json.load(open(f"{a.dir}/shapes.json"))["rows"]}
    if a.layers:
        want = []
        for part in a.layers.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                want.extend(range(int(lo), int(hi) + 1))
            else:
                want.append(int(part))
    else:
        want = sorted(rows)

    lines = [f"# stream O: {len(want)} layer chunks from {a.dir}"]
    for k in want:
        tb = rows[k]["table_bits"]
        lines.append(f"--table-bits {tb} --bounds "
                     f"--witness {a.dir}/witness_L{k} --ser-tag L{k}")
    if a.head:
        htb = json.load(open(f"{a.head}/manifest.json"))["master_table_bits"]
        lines.append("# the head, last, and it is the same compose family with two extra "
                     "arguments")
        lines.append(f"--table-bits {htb} --head-argmax --deq-per-row 501 "
                     f"--witness {a.head} --ser-tag head")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {sum(1 for l in lines if not l.startswith('#'))} chunks")


if __name__ == "__main__":
    main()
