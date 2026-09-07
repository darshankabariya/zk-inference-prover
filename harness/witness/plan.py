#!/usr/bin/env python3
"""Write the segment loop's plan file for `zk-prover compose-chunks --plan`.

One line per chunk, in the order the sequence proves them: the 32 layers and
then the head. A line carries only what that chunk decides for itself -- its
witness, its master table, and for the head the argmax and its row count.
Everything the whole sequence shares (the round, the band sizes, the pins, the
tight table) stays on the command line, because that is where the loop's
shared arguments live and repeating them per line is how a driver drifts.

The lines are the same arguments `n_seq.sh`'s `seq` step passes to 33 separate
`compose` processes, which is what makes the loop's digests comparable to
theirs chunk by chunk.

**Stream DD (E11): the block axis.**  A full prompt+output proof is `B` token
blocks of 32 layer chunks, plus one `lm_head` chunk, and the writer learns that with
one extra loop and no new argument *kind*:

    python -m harness.witness.plan --dir D --blocks 9 --head D/witness_B8_head --out plan.txt

reads `D/shapes_B<b>.json` and writes `--witness D/witness_B<b>_L<k>
--ser-tag B<b>L<k>`.  What a prompt block's line carries that a decode line
does not is **nothing new** -- the prefix length is `Params.c`, which
`params_of` reads off the witness's own manifest (`main.rs:567`,
`c: b0.shape.c`), so a block is a directory and a table size and that is all.
No `Params` field is added, which matters because `boundary.rs:253-257`'s
`shape_ok` compares `params.s`, `params.glue`, the blowup and the two boundary
trees' sizes and would reject every cross-block link if a block index entered
the statement.

Two block-axis facts the writer enforces rather than leaves to the runner:

* **Block 0 has no prefix.**  `c = 0` makes `Grid::new(groups = c * insts, ..)`
  divide by zero (`layout.rs:117`), so a corner-only chunk does not compose
  today -- that configuration is stream CC's.  Block 0 is therefore skipped
  unless `--with-block-0` is passed, and the plan says so in a comment.
* **The head goes on the last block only.**  A prompt block emits no token.
  `--head-every-block` is there for the variant that proves one per block, and
  it is off, because nothing downstream asks for it yet.

`--emit-out` writes the *emit* set: one representative line per distinct
constraint-kernel shape rather than one per chunk.  `emitset` runs one
`cudaemit` per plan line and the store dedups by fingerprint afterwards, so at
257 lines it launches 257 processes to install ~40 shapes.  Within one model
and one set of shared pins the only shape-bearing things that vary down a plan
are the prefix length and the master table size, so `(c, table_bits)` -- plus
the head's two extra arguments -- is the whole distinct set.

**Stream EE (E11): the flags a line carries.**  `--layer-flags` goes on every
layer line, `--l0-flags` on the layer-0 lines after it, `--head-flags` on the
head line, and `--l0-dir` points the layer-0 lines at another dump set (a
layer-0 re-extract with `--embed`).  The E11 plan at C=4096 is

    python -m harness.witness.plan --dir D --blocks 9 --with-block-0 --head H \
        --layer-flags='--round even --corner' --head-flags='--ao-rem-bits 23'

(the `=` form, because the value starts with a dash), and a line's flags are
part of its emit key, so the emit set grows by the lines whose flags make a
new kernel shape and by nothing else.

**Stream MM (E11): what the model manifest needs of a plan.**  `zk-prover
model-roots` requires `master_table_bits` to be **one value per layer across
blocks** -- `table/exp` is `1 << log_table` rows, so a layer whose blocks
declare two widths has two tree-0 roots and no per-layer model root exists at
all.  Each block's shape table gives that block's own tightest width, and on
the recorded C=4096 extraction seven of the 33 layers carry two, so the
recorded plan is rejected (stream LL's open item 3).  `--table-bits-per-layer`
writes, for every block of a layer, the widest width any block of that layer
asked for; the extra rows are the price of a manifest, and the plan must then
run **without `--tight-table`**, which re-sizes the table off the witness
after the flag is read and which no manifest and no flag names.

`--head-per-block` says the head dump for block `b` is `<--head>_B<b>` rather
than `--head` itself: a head dump carries its own `base_pos`, so the two seam
blocks a `--verify --prompt` grid needs (stream KK) are two dumps, written by
one `extract.py --blocks 7,8 --head --corner` pass (stream OO).
"""

import argparse
import json
import os


def read_rows(path):
    return {r["layer"]: r for r in json.load(open(path))["rows"]}


def want_layers(spec, rows):
    if not spec:
        return sorted(rows)
    want = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            want.extend(range(int(lo), int(hi) + 1))
        else:
            want.append(int(part))
    return want


def layer_line(tb, wdir, tag, extra=""):
    return f"--table-bits {tb}{extra} --witness {wdir} --ser-tag {tag}"


def head_line(hdir, htb, tag="head", extra="", layer=None, block=None):
    """The `lm_head` line.

    **Stream KK (E11): the coordinate.**  A head dump's manifest says which
    layer it was *extracted* at (16, for the C=4K set) and carries no block at
    all, but the grid puts the head chunk after the last layer of the block it
    reads.  `--layer` / `--block` declare that, `Fixed` carries it, and
    `boundary-grid` places the chunk by the statement rather than by the tag --
    which is what lets the head take a cell at all.

    **Stream UU (E11): the residual boundary.**  The head chunk commits the residual its
    final norm consumed as tree 3, exactly as a layer chunk commits its
    `stream_in`, so the last layer's out root and the head's in root are
    compared by the pair check like every other layer link.  It commits no tree
    4 -- its norm's output is nobody's residual -- so the head is the last chunk
    of the layer axis and never a producer.  A head dump too old to carry the
    final norm has no wiring family and commits nothing here; the grid reports
    that as an unchecked end rather than as a link that failed.
    """
    coord = ""
    if layer is not None:
        coord += f" --layer {layer}"
    if block is not None:
        coord += f" --block {block}"
    return (f"--table-bits {htb} --deq-per-row 501{extra}{coord} "
            f"--witness {hdir} --ser-tag {tag}")


def flags(s):
    """A flag string as written on a line: one leading space, or nothing."""
    s = " ".join(s.split())
    return f" {s}" if s else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="the layer witness directory, holding shapes.json and witness_L*")
    ap.add_argument("--head", default="", help="the head witness directory, if it is to be proved")
    ap.add_argument("--layers", default="", help="which layers, e.g. 0-31 or 0,1,2 (default all)")
    ap.add_argument("--out", required=True)
    # ---- stream DD (E11): the block axis ------------------------------------
    ap.add_argument("--blocks", default="",
                    help="how many token blocks the sequence has: an integer, or 'auto' to "
                         "count DIR/shapes_B*.json. Empty (the default) is the pre-DD "
                         "single-block plan and its lines are unchanged.")
    ap.add_argument("--with-block-0", action="store_true",
                    help="include block 0's layer chunks. It has c = 0 and does not compose "
                         "today (layout.rs:117 divides by c * insts); stream CC owns the "
                         "corner-only configuration.")
    ap.add_argument("--head-every-block", action="store_true",
                    help="an lm_head chunk after every block, not only the last one")
    # ---- stream KK (E11): which blocks need a head chunk, and why ------------
    ap.add_argument("--head-blocks", default="",
                    help="the blocks that get an lm_head chunk, e.g. '7,8' or '0-8' "
                         "(default: the last block; --head-every-block is '0-B'). "
                         "A GENERATED position's id has to equal the argmax the head "
                         "proved for the position before it, so the block that owns the "
                         "position before the first generated one needs a head chunk too "
                         "-- for the 9-block C=4K sequence (8 prompt blocks + 1 decode "
                         "block) that is block 7, and '--head-blocks 7,8' is what makes "
                         "`boundary-grid --verify` report no unchecked generated position")
    ap.add_argument("--emit-out", default="",
                    help="also write the distinct emit shapes, one representative plan line "
                         "each, for `emitset` to walk instead of the plan")
    # ---- stream EE (E11): what every line carries besides its own shape ------
    # The E11 grid wants `--round even --corner` on every layer line (the
    # boundaries and the embedding gadget are on by default wherever the dump
    # lets them be) and the head pinned where seq6 ran it.  They go on the LINE,
    # not the shared command line, because `boundary-grid` reads a chunk's flags
    # back off its statement and the head must not inherit `--corner`.  A line's
    # flags are part of its emit key, and so is what the dump lets the defaults
    # turn on: a layer-0 line whose dump carries the embedding rows is a
    # different kernel shape from a layer-1 line at the same (c, table bits).
    ap.add_argument("--layer-flags", default="",
                    help="appended to every layer line, e.g. "
                         "'--round even --corner' (E11)")
    ap.add_argument("--l0-flags", default="",
                    help="appended to every layer-0 line after --layer-flags, e.g. "
                         "(E11 needs none: the embedding gadget is on by default on a "
                         "layer-0 dump that carries embed.u16 and tok_in.i32)")
    ap.add_argument("--l0-dir", default="",
                    help="take the layer-0 witnesses from this directory instead of --dir "
                         "(a layer-0 re-extract with --embed next to the full dump set)")
    ap.add_argument("--head-flags", default="",
                    help="appended to the head line, e.g. '--ao-rem-bits 23' to keep the "
                         "head at its own pin when the shared pins are the layers'")
    # ---- stream SS (E11): the attention-output remainder, per line ----------
    # A shared `--ao-rem-bits` is the model-wide worst case, and a chunk whose
    # own `ao_q_bits` is wide refuses it: `compose.rs` asserts
    # `ao_q_bits + ao_rem_bits <= 30`, so the E11 set's `attn_out_bits 12`
    # (`ao_q_bits` 11) refuses the shared 20 that its own worst block needs,
    # while its own 19 is legal.  A line's arguments are placed BEFORE the
    # shared ones (`chunks.rs::specs`) and `arg_scalar` takes the first match,
    # so a per-line pin simply wins and the shared one stays as the default for
    # the lines that do not carry one.  Off by default: with it absent every
    # plan this repo has recorded is byte-identical.
    ap.add_argument("--ao-rem-per-line", action="store_true",
                    help="write each layer line's own --ao-rem-bits, off the shape table's "
                         "row, instead of leaving it to the shared pin")
    # ---- stream MM (E11): one master table width per layer ------------------
    ap.add_argument("--table-bits-per-layer", action="store_true",
                    help="give every block of a layer the widest --table-bits any block of "
                         "that layer asked for, so `zk-prover model-roots` can derive one model "
                         "root per layer (it rejects a layer whose blocks declare two "
                         "widths). Run the plan WITHOUT --tight-table, which re-sizes the "
                         "table off the witness and which no manifest names.")
    ap.add_argument("--head-per-block", action="store_true",
                    help="the head dump for block b is <--head>_B<b>, not --head itself "
                         "(one `extract.py --blocks 7,8 --head --corner` pass writes both)")
    ap.add_argument("--table-bits-map", default="",
                    help="a JSON {\"<layer>\": bits, ...} (and optionally \"head\") giving "
                         "each layer's stated master table width, instead of the shape "
                         "table's. The shape table carries the EXTRACTION's "
                         "`master_table_bits`, which stream J measured 1-3 bits loose, and "
                         "declaring that without --tight-table put the C=4096 grid 137.9 GB "
                         "into a 150.1 GB card (stream MM). A map built from a previous "
                         "run's own tightened `params.log_table`, maxed over the blocks of "
                         "each layer, is both tight and one width per layer, which is what "
                         "`zk-prover model-roots` needs.")
    a = ap.parse_args()
    lf, l0f, hf = flags(a.layer_flags), flags(a.l0_flags), flags(a.head_flags)

    if not a.blocks:
        blocks = [None]
    elif a.blocks == "auto":
        import glob
        blocks = sorted(int(os.path.basename(q)[len("shapes_B"):-len(".json")])
                        for q in glob.glob(f"{a.dir}/shapes_B*.json"))
        assert blocks, f"--blocks auto: no {a.dir}/shapes_B*.json"
    else:
        blocks = list(range(int(a.blocks)))

    # stream KK: which blocks carry a head chunk.  `None` keeps the pre-KK rule
    # (the last block, tagged `head`); a list tags them `B<b>head` unless there
    # is only one.
    if a.head_blocks:
        head_blocks = set(want_layers(a.head_blocks, {b: None for b in blocks if b is not None}))
    elif a.head_every_block:
        head_blocks = {b for b in blocks if b is not None}
    else:
        head_blocks = None
    # the layer coordinate the head chunk sits at: one past the last layer the
    # plan writes, read off the dump's own shapes rather than assumed to be 32
    n_layers = 1 + max(read_rows(f"{a.dir}/shapes.json" if blocks == [None]
                                 else f"{a.dir}/shapes_B{blocks[-1]}.json"))

    # stream MM: the widest `table_bits` any block of a layer asked for, so one
    # model root per layer exists.  Built over the blocks the plan actually
    # writes -- a width from a block this plan skips is not this plan's.
    widest = {}
    if a.table_bits_map:
        m = json.load(open(a.table_bits_map))
        widest = {int(k): int(v) for k, v in m.items() if k != "head"}
    elif a.table_bits_per_layer:
        for b in blocks:
            if b is None:
                rows_b = read_rows(f"{a.dir}/shapes.json")
            else:
                if b == 0 and not a.with_block_0:
                    continue
                rows_b = read_rows(f"{a.dir}/shapes_B{b}.json")
            for k, r in rows_b.items():
                widest[k] = max(widest.get(k, 0), r["table_bits"])

    lines, emit = [], {}
    n_rows, n_layer_rows = 0, 0
    for b in blocks:
        if b is None:
            rows = read_rows(f"{a.dir}/shapes.json")
            pre, wpre = "", f"{a.dir}/witness_L"
        else:
            if b == 0 and not a.with_block_0:
                lines.append("# block 0 skipped: c = 0 is the corner-only chunk (stream CC); "
                             "pass --with-block-0 once it composes")
                continue
            rows = read_rows(f"{a.dir}/shapes_B{b}.json")
            pre, wpre = f"B{b}", f"{a.dir}/witness_B{b}_L"
            lines.append(f"# block {b}: prefix c = {rows[sorted(rows)[0]]['c']}")
        want = want_layers(a.layers, rows)
        for k in want:
            tb = widest.get(k, rows[k]["table_bits"])
            extra = lf + (l0f if k == 0 else "")
            if a.ao_rem_per_line:
                rem = rows[k].get("ao_rem_bits")
                assert rem, f"--ao-rem-per-line: {a.dir}'s shape table has no ao_rem_bits"
                extra += f" --ao-rem-bits {rem}"
            wdir = f"{wpre}{k}"
            if k == 0 and a.l0_dir:
                wdir = f"{a.l0_dir}/{os.path.basename(wpre)}{k}"
            line = layer_line(tb, wdir, f"{pre}L{k}", extra)
            lines.append(line)
            n_rows += 1
            n_layer_rows += 1
            emit.setdefault((rows[k]["c"], tb, False, extra), line)
        if a.head and (b in head_blocks if head_blocks is not None else b == blocks[-1]):
            hdir = f"{a.head}_B{b}" if a.head_per_block else a.head
            htb = json.load(open(f"{hdir}/manifest.json"))["master_table_bits"]
            if a.table_bits_map:
                hm = json.load(open(a.table_bits_map)).get("head")
                if hm is not None:
                    htb = int(hm)
            elif a.table_bits_per_layer and a.head_per_block:
                # the head chunks sit at one layer coordinate too, so they need
                # one width between them for the same reason the layers do
                htb = max(json.load(open(f"{a.head}_B{hb}/manifest.json"))
                          ["master_table_bits"] for hb in sorted(head_blocks))
            lines.append("# the head, last, and it is the same compose family with two extra "
                         "arguments")
            solo = head_blocks is None or len(head_blocks) == 1
            hl = head_line(hdir, htb, "head" if solo else f"{pre}head", hf,
                           layer=n_layers, block=b)
            lines.append(hl)
            n_rows += 1
            emit.setdefault((None, htb, True, hf), hl)
    lines.insert(0, f"# stream O: {n_layer_rows} layer chunks from {a.dir}"
                    if blocks == [None] else
                    f"# stream DD: {n_layer_rows} layer chunks over {len(blocks)} token "
                    f"blocks from {a.dir}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {n_rows} chunks")

    if a.emit_out:
        el = [f"# stream DD: {len(emit)} distinct emit shapes for the {n_rows} chunks of "
              f"{a.out}; the key is (prefix c, master table bits, head, the line's flags)"]
        for key in sorted(emit, key=lambda k: (k[2], k[0] or 0, k[1], k[3])):
            el.append(emit[key])
        os.makedirs(os.path.dirname(a.emit_out) or ".", exist_ok=True)
        with open(a.emit_out, "w") as f:
            f.write("\n".join(el) + "\n")
        print(f"{a.emit_out}: {len(emit)} emit shapes (of {n_rows} chunks, "
              f"x{n_rows / max(len(emit), 1):.1f} fewer cudaemit launches)")


if __name__ == "__main__":
    main()
