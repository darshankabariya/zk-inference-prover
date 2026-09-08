"""The model-wide shape table, read off the layer witnesses' manifests.

A chunk loop over a real model may mix table sizes but not two statements
(`prover/src/chunks.rs`), so `kmax` and `ao_rem_bits` are pinned at the
model-wide worst case while `master_table_bits` varies per layer.  This reads
what each layer needs -- before a loop is launched, not from the assert it
trips -- and prints the table B10's record carries.

`ao_rem_bits` is `Layer::rem_bits()` in `prover/src/layer.rs`:

    bits( 2 * max_{h,t,j} ( Z[h,t] << max(-(jv[h,j] - jo[t]), 0) ) )

which for a fixed (h, t) is maximised at the smallest `jv[h, .]`, so the whole
maximum is over (h, t) alone.

    uv run python -m harness.witness.shapes <witness base> 32 [<out.json>]

**Stream DD (E11): the block axis.**  A sequence has one of these per token
block, so the output path is no longer derivable from the base alone -- every
block's base lives in the same directory -- and the *model-wide* pins are no
longer a maximum over 32 layers but over `B x 32`.  Two additions, both
backwards compatible:

    ... shapes <base> <n> [<out.json>]        one block, named output
    ... shapes --merge a.json b.json ... --out shapes_all.json

The merge recomputes `worst` over every row of every input -- `ao_rem_bits`
grows with the prefix, so the last block's is the model-wide one and a plan
pinned from block 0 would be rejected by `params_of`'s
`--ao-rem-bits {want} is below this witness's {}` assertion at the last chunk,
after hours of proving.

**Issue #118: the bound the pin has to leave room for.**  `ao_rem_bits` is one
number for the whole loop and every layer is proved at it, so the model-wide
worst case can push a layer past the quotient cap the attention output's
rounding identity has to fit inside -- `q_bits + rem_bits`, against 30 in the
plain form and 31 in the `denp2` form (`Plan::new`) -- even though that layer's
own divisor range fits.  `worst_of` therefore evaluates that bound, in both
forms, on every row it took the pin over, and refuses the pin with the layer and
the numbers rather than leaving the refusal to the first chunk of a loop that is
hours old.  Every table also prints the tightest window and what is left of it:
the recorded C=4,096 set sits at `11 + 20 = 31` of 31, which composes and admits
nothing wider.  The layer's own value is legal, so what the refusal names is
`plan.py --ao-rem-per-line`, which writes it on that layer's own line.

    ... shapes --selftest                     the bound and its refusal, no dump
"""
import argparse
import json
import os
import sys


# ---- the attention output's no-wrap bound (issues #111, #118) ---------------
# The rounding identity the attention-output fold proves multiplies `2 mag num`,
# and it is an identity over the integers only while that product stays under
# the field's `2^31 - 1`.  `prover/src/main.rs`'s `params_of` sizes the two
# accumulator limbs, picks each form off the numbers it is given, and
# `Plan::new` (`prover/src/compose.rs`) refuses a statement whose window does
# not fit -- at proving time, hours after the extraction and the plan that
# decided it.  The functions below are that arithmetic, so `extract.py` refuses
# a dump that would not compose and this table refuses a pin that would not.
# They live here because this module imports neither torch nor numpy at import
# time, and the plan writer does not want either.
#
# `MIN_LOG_ROWS` is `prover/src/common.rs`: a range table is its own `1 << bits`
# rows and the LogUp builder has no shorter column, so no limb is narrower.
MIN_LOG_ROWS = 4
# `p = 2^31 - 1`, so the product has 31 bits to fit in.  The sibling bound's cap
# is 30 in the plain form and the same 31 in the `denp2` form.
NO_WRAP_BITS = 31


def ao_acc_split(need):
    """`params_of`'s two accumulator limbs for a witness needing `need` bits.

    The low limb takes what it can up to 16, the widest range table the layer
    already builds; the high limb takes the rest; neither goes below the
    shortest column the builder has.
    """
    lo = min(max(need, MIN_LOG_ROWS), 16)
    return lo, max(need - lo, MIN_LOG_ROWS)


def ao_quotient(attn_out_bits, rem_bits):
    """The no-wrap bound's sibling term `q_bits + rem_bits`, in its two forms.

    `q_bits` is the fold's target width minus one, which is the dump's own, and
    `rem_bits` is the divisor range the chunk is proved at -- its own, or the
    wider one a plan pins it at, and the only one of the two a plan can move.
    The plain form's cap is 30 and the `denp2` form's is 31; `params_of` takes
    the `denp2` form exactly when the plain form no longer fits.
    """
    q_bits = attn_out_bits - 1
    used = q_bits + rem_bits
    denp2 = used > 30
    limit = NO_WRAP_BITS if denp2 else 30
    return dict(bits=used, limit=limit, margin=limit - used,
                q_bits=q_bits, rem_bits=rem_bits,
                form="denp2" if denp2 else "plain")


def ao_no_wrap(acc_bits, jv_max, jo_min, attn_out_bits, rem_bits):
    """The attention output's no-wrap bound, in the two forms the prover uses.

    The accumulator's two limbs bound one factor of `2 mag num` and the exponent
    table's own domain bounds the other, so what has to hold is
    `1 + lo + hi + e_hi <= 31` with `e_hi = jv_max - jo_min`.  The offset form
    spends the accumulator's sign bit on the offset and the magnitude form
    spends it on the window instead, which is worth one bit and no more;
    `params_of` takes the magnitude form exactly when the offset form does not
    fit, so that every dump that composed before it existed keeps its columns.

    The sibling bound of the same identity is `ao_quotient` above, and it is the
    half a plan's pins move.

    Returns what each window comes to, its limit and its **margin**: how many
    bits a wider context, another model or a different quantization may add
    before this layer stops composing.  A negative margin is a dump that would
    not prove.
    """
    e_hi = max(jv_max - jo_min, 0)
    lo_off, hi_off = ao_acc_split(acc_bits + 1)
    lo_mag, hi_mag = ao_acc_split(acc_bits)
    mag = 1 + lo_off + hi_off + e_hi > NO_WRAP_BITS
    lo, hi = (lo_mag, hi_mag) if mag else (lo_off, hi_off)
    bits = 1 + lo + hi + e_hi
    return dict(
        bits=bits, limit=NO_WRAP_BITS, margin=NO_WRAP_BITS - bits,
        acc_bits=acc_bits, acc_lo_bits=lo, acc_hi_bits=hi, e_hi=e_hi,
        form="magnitude" if mag else "offset",
        quotient=ao_quotient(attn_out_bits, rem_bits),
    )


def ao_quotient_refusal(q, where):
    """What `where`'s quotient term is refused with, or `None` if it fits."""
    if q["margin"] >= 0:
        return None
    return (f"{where}: the attention output's quotient term does not fit: "
            f"{q['q_bits']} + {q['rem_bits']} = {q['bits']} > {q['limit']} (over by "
            f"{-q['margin']}, {q['form']} form). `q_bits` is the fold's target width "
            f"minus one and `rem_bits` is the divisor range this chunk is proved at, "
            f"and the one bit the `denp2` form is worth is already taken.")


def ao_no_wrap_refusal(w, where):
    """What `where`'s window is refused with, or `None` if it fits.

    The message carries the numbers, because there is nothing to widen: the two
    limbs must together cover `|acc|` however many of them there are, and `e_hi`
    is the exponent table's own domain, so what is left is a narrower shape.
    """
    if w["margin"] < 0:
        return (f"{where}: the attention output's no-wrap window does not fit: "
                f"1 + {w['acc_lo_bits']} + {w['acc_hi_bits']} + {w['e_hi']} = {w['bits']} "
                f"> {w['limit']} (over by {-w['margin']}, {w['form']} form). The two limbs "
                f"must together cover |acc| ({w['acc_bits']} bits) and the exponent window "
                f"must cover e = jv - jo, so no wider decomposition of either recovers a "
                f"bit, and the one bit the magnitude form is worth is already taken. What "
                f"is left is a narrower context or a quantization whose accumulator is "
                f"narrower.")
    return ao_quotient_refusal(w["quotient"], where)


# ---- issue #118: the bound under the pins this table writes -----------------
# `ao_rem_bits` below is a model-wide maximum, and it goes on the chunk loop's
# command line, so every layer of every block is proved at it.  A layer whose
# own `attn_out_bits` is wide can be pushed past the quotient cap by that pin
# even though its own divisor range fits, and until this check existed the
# refusal came from `Plan::new` at the first chunk of a loop that was hours old.
# The layer's own value is legal, so the fix is a per-line pin, which is what
# the message says: `plan.py --ao-rem-per-line`.


def ao_where(r):
    """A row, named as a refusal names it."""
    return (f"layer {r['layer']}" if r.get("block") is None
            else f"block {r['block']} layer {r['layer']}")


def ao_pin(rows):
    """The `--ao-rem-bits` a chunk loop over `rows` is pinned at.

    `params_of` refuses a pin below the witness's own (`--ao-rem-bits {want} is
    below this witness's {}`), so the model-wide pin is the maximum over every
    row, which is what `worst_of` writes and what the drivers read.
    """
    return max(r["ao_rem_bits"] for r in rows)


def ao_pin_row(r, rem_bits):
    """One row's quotient window at `rem_bits`, and what it is refused with.

    The refusal names the layer and the numbers, and then says which of the two
    it is: a pin this layer cannot take although its own divisor range fits, or
    a shape that does not compose at any pin.
    """
    q = ao_quotient(r.get("attn_out_bits", 8), rem_bits)
    msg = ao_quotient_refusal(q, ao_where(r))
    if msg is None:
        return q, None
    own = ao_quotient(r.get("attn_out_bits", 8), r["ao_rem_bits"])
    if own["margin"] >= 0:
        msg += (f" This layer's own divisor range is {own['rem_bits']} bits and does fit "
                f"({own['bits']}/{own['limit']}, margin {own['margin']}, {own['form']} "
                f"form), so what does not fit is the model-wide pin. A line's arguments "
                f"are read after the shared ones and the last one wins, so "
                f"`plan.py --ao-rem-per-line` writes this layer's own value on this "
                f"layer's own line and the loop keeps the shared pin for the rest.")
    else:
        msg += (f" This layer's own divisor range is {own['rem_bits']} bits and does not "
                f"fit either ({own['bits']}/{own['limit']}), so no pin recovers it: what "
                f"is left is a narrower context or a quantization whose divisor range is "
                f"narrower.")
    return q, msg


def ao_pin_check(rows, rem_bits):
    """Every row at `rem_bits`: the tightest row, its window, and the refusals."""
    seen = [(r,) + ao_pin_row(r, rem_bits) for r in rows]
    tight = min(seen, key=lambda t: t[1]["margin"])
    return tight[0], tight[1], [m for _, _, m in seen if m]


def ao_pin_refusals(bad):
    """The refusals as one message: three of them, and then how many more.

    Every layer of a set is refused by one pin for the same reason, and the
    three that are printed carry the numbers of all of them.
    """
    return "\n\n".join(bad[:3] + ([f"... and {len(bad) - 3} more layers, at the same pin"]
                                   if len(bad) > 3 else []))


def ao_pin_report(rows, rem_bits):
    """The line the table prints about the pin it writes: the tightest window.

    A margin of zero is a pin that composes and admits nothing wider, which is
    where the recorded C=4,096 set sits: `11 + 20 = 31` in the `denp2` form.
    """
    r, q, _ = ao_pin_check(rows, rem_bits)
    tail = ("; a wider context, another model or a different quantization adds a bit "
            "and this layer stops composing" if q["margin"] == 0 else "")
    return (f"attention output: at --ao-rem-bits {rem_bits} the tightest quotient window "
            f"is {q['bits']}/{q['limit']} bits, margin {q['margin']} ({q['form']} form), "
            f"at {ao_where(r)}{tail}")


def shape_of(d):
    # Stream DD: imported here rather than at the top so the merge mode, which
    # reads JSON and nothing else, runs where numpy is not installed (the desk
    # self-test at `b12/e11/dd_selftest.py`).  It is cached after the first of
    # the 32 calls.
    import numpy as np

    man = json.load(open(os.path.join(d, "manifest.json")))
    nh, s = man["heads"], man["s"]
    zz = np.fromfile(os.path.join(d, "zz.i32"), dtype=np.int32).reshape(nh, s).astype(np.int64)
    jo = np.fromfile(os.path.join(d, "jo.i32"), dtype=np.int32).astype(np.int64)      # (S,)
    jv = np.fromfile(os.path.join(d, "jv.i32"), dtype=np.int32).reshape(nh, -1)        # (nh, hd)
    shift = np.maximum(jo[None, :] - jv.min(-1)[:, None], 0)                           # (nh, S)
    rem = int(2 * (zz << shift).max())
    accs = man["accumulators"]
    # stream Z: a dump extracted with a fold matmul carries the fold's own
    # divisor range; the model-wide pins come off these exactly as `ao_rem_bits`
    fold = man.get("fold") or {}
    return dict(
        layer=man["layer"], c=man["c"], s=man["s"],
        # Stream DD: absent on a pre-DD dump, and `None` then rather than 0 --
        # a decode dump is not "block 0".
        block=man.get("block"),
        table_bits=man["master_table_bits"], spread=man["frac_spread"],
        d_bits=man["d_bits"], z_max=man["z_max"],
        ao_rem_bits=int(rem).bit_length(),
        attn_out_bits=man.get("attn_out_bits", 8),
        kmax_needed=max(v["kmax_needed"] for v in accs.values()),
        kmax_blind=max(v["kmax_worst_case"] for v in accs.values()),
        fold_q_max=man["attn_out"]["fold_q_max"],
        fold_rem_bits=fold.get("fold_rem_bits", 0),
        fold_gt_min=fold.get("gt_min", 0), fold_gc_min=fold.get("gc_min", 0),
        slim=man.get("slim", False),
    )


def worst_of(rows):
    """The model-wide pins a chunk loop has to be launched with.

    Stream DD: taken over whatever rows it is given, which is one block's 32
    layers in the pre-DD call and `B x 32` under the merge mode.
    """
    tb = sorted({r["table_bits"] for r in rows})
    hist = {b: sum(1 for r in rows if r["table_bits"] == b) for b in tb}
    worst = dict(ao_rem_bits=ao_pin(rows),
                 kmax=max(r["kmax_blind"] for r in rows),
                 fold_q_max=max(r["fold_q_max"] for r in rows),
                 table_bits=hist)
    # stream Z: the fold's model-wide pins.  `fold_rem_bits` is a plain maximum;
    # `fold_alpha` must sit under every layer's min G_t while covering every
    # layer's min G_c, which is a joint condition and is asserted, not assumed.
    if any(r["fold_rem_bits"] for r in rows):
        alpha = max(max(14 - r["fold_gc_min"], 0) for r in rows if r["fold_rem_bits"])
        gt_lo = min(r["fold_gt_min"] for r in rows if r["fold_rem_bits"])
        assert alpha <= gt_lo, f"fold: no alpha fits (need <= {gt_lo}, want {alpha})"
        worst["fold_rem_bits"] = max(r["fold_rem_bits"] for r in rows)
        worst["fold_alpha"] = alpha
    # stream Z: the attnout fold's quotient clamps at the dump's own width
    assert all(r["fold_q_max"] <= (1 << (r["attn_out_bits"] - 1)) - 1 for r in rows), \
        "a layer's fold quotient is past its clamp"
    # issue #118: and the bound the pin above has to leave room for, on every
    # row it was taken over.  `ao_rem_bits` is one number for the whole loop, so
    # a layer is proved at the model-wide worst case and not at its own; a layer
    # that does not fit it is refused here rather than by `Plan::new`, hours in.
    if all(r.get("ao_rem_bits") for r in rows):
        _, _, bad = ao_pin_check(rows, worst["ao_rem_bits"])
        assert not bad, ao_pin_refusals(bad)
    return worst, tb


def _row(layer, rem, bits=12, block=None, table_bits=23):
    """A shape table row with the fields the pins are taken over."""
    return dict(layer=layer, block=block, c=512, s=512, table_bits=table_bits,
                spread=0.0, d_bits=16, z_max=1, ao_rem_bits=rem, attn_out_bits=bits,
                kmax_needed=19, kmax_blind=19, fold_q_max=(1 << (bits - 1)) - 1,
                fold_rem_bits=15, fold_gt_min=8, fold_gc_min=8, slim=True)


def selftest():
    """Issue #118: the bound under the pin this table writes.

    The numbers are the recorded C=4,096 set's: `attn_out_bits` 12, so `q_bits`
    11, and divisor ranges reaching the 20 whose `denp2` window is the field's
    31 exactly.  `plan.py --selftest` checks the same bound where a plan line
    resolves its pin; this checks it where the pin is decided.
    """
    bad = 0

    def case(label, ok, note):
        nonlocal bad
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:<52s} -> {note}")

    # the two forms, at the two fold widths the sets use
    got = [(ao_quotient(8, 23)["form"], ao_quotient(8, 23)["margin"]),
           (ao_quotient(8, 24)["form"], ao_quotient(8, 24)["margin"]),
           (ao_quotient(12, 20)["form"], ao_quotient(12, 20)["margin"]),
           (ao_quotient(12, 21)["form"], ao_quotient(12, 21)["margin"])]
    want = [("plain", 0), ("denp2", 0), ("denp2", 0), ("denp2", -1)]
    case("the quotient term's two forms and their margins", got == want, got)

    # the recorded set: the pin is the maximum over every row and it composes
    rows = [_row(k, r, block=b) for b in (0, 1)
            for k, r in enumerate((14, 15, 19) if b == 0 else (15, 16, 20))]
    worst, _ = worst_of(rows)
    _, q, refusals = ao_pin_check(rows, worst["ao_rem_bits"])
    ok = (worst["ao_rem_bits"] == 20 and not refusals and q["bits"] == 31
          and q["margin"] == 0 and q["form"] == "denp2")
    case("the recorded pin, at the denp2 form's cap exactly", ok,
         ao_pin_report(rows, worst["ao_rem_bits"]))

    # and the layer this check exists for: one row extracted at the narrower
    # fold width carries a wider divisor range, so the pin it sets is one no
    # 12-bit row can take, although every one of them takes its own
    rows[-1] = _row(2, 23, bits=8, block=1)
    try:
        worst_of(rows)
        got = "NOT refused"
    except AssertionError as e:
        got = str(e).splitlines()[0]
    ok = (got.startswith("block 0 layer 0: ")
          and "11 + 23 = 34 > 31 (over by 3, denp2 form)" in got
          and "--ao-rem-per-line" in got)
    case("a pin no layer of the set can take is refused", ok, got.split(".")[0])

    # a table whose rows are one block's, as the pre-DD call writes it: the
    # refusal names the layer and no block, because the row carries none
    one = [_row(k, r) for k, r in enumerate((15, 16, 21))]
    try:
        worst_of(one)
        got = ["NOT refused"]
    except AssertionError as e:
        got = str(e).split("\n\n")
    ok = (len(got) == 3 and got[0].startswith("layer 0: ")
          and "--ao-rem-per-line" in got[0]
          and got[-1].startswith("layer 2: ") and "no pin recovers it" in got[-1])
    case("one block's table names the layer, and each case", ok, got[0].split(".")[0])

    return 1 if bad else 0


def merge_tables(paths, out):
    """Stream DD: one `worst` over several blocks' shape tables."""
    rows = []
    for q in paths:
        d = json.load(open(q))
        for r in d["rows"]:
            r = dict(r)
            if r.get("block") is None:
                r["block"] = d.get("block")
            rows.append(r)
    worst, _ = worst_of(rows)
    blocks = sorted({r.get("block") for r in rows}, key=lambda b: (b is None, b))
    json.dump(dict(rows=rows, worst=worst, merged_from=list(paths), blocks=blocks),
              open(out, "w"), indent=1)
    print(f"merged {len(paths)} tables, {len(rows)} rows, blocks {blocks}")
    print(f"worst case: {json.dumps(worst)}")
    print(ao_pin_report(rows, worst["ao_rem_bits"]))
    print(f"-> {out}")


if __name__ == "__main__":
    # issue #118: the bound's own test, read before the parser because the
    # positional arguments below are a witness set and this needs none
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    # Stream DD: the merge mode folds several blocks' tables into the one
    # `worst` a sequence's chunk loop has to be pinned from.
    if "--merge" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--merge", nargs="+", required=True)
        ap.add_argument("--out", required=True)
        aa = ap.parse_args()
        merge_tables(aa.merge, aa.out)
        sys.exit(0)
    base, n = sys.argv[1], int(sys.argv[2])
    rows = [shape_of(f"{base}{k}") for k in range(n)]
    print(f"{'L':>3} {'C':>6} {'S':>4} {'tbl':>4} {'sprd':>5} {'dbits':>6} {'zmax':>8} "
          f"{'aorem':>6} {'kmax':>5} {'blind':>6} {'foldq':>6}")
    for r in rows:
        print(f"{r['layer']:>3} {r['c']:>6} {r['s']:>4} {r['table_bits']:>4} "
              f"{r['spread']:>5.1f} {r['d_bits']:>6} {r['z_max']:>8} {r['ao_rem_bits']:>6} "
              f"{r['kmax_needed']:>5} {r['kmax_blind']:>6} {r['fold_q_max']:>6}")
    worst, tb = worst_of(rows)
    # 5.010 fresh cells per master-table row (B9 1.5), so what the tables cost
    # per-layer against forcing the model's largest on all of them
    per = {b: 5.010 * (1 << b) for b in tb}
    own = sum(per[r["table_bits"]] for r in rows)
    one = per[max(tb)] * len(rows)
    print(f"\nworst case: {json.dumps(worst)}")
    print(ao_pin_report(rows, worst["ao_rem_bits"]))
    print(f"master table, per-layer {own / 1e9:.3f} G cells vs one shape at 2^{max(tb)} "
          f"{one / 1e9:.3f} G = x{one / own:.2f}")
    # Stream DD: the output path.  Pre-DD it was `<dir>/shapes.json` and with two
    # arguments it still is; a sequence passes a third, because every block's
    # base lives in the same directory and the default would have nine blocks
    # overwriting one file.
    out = (sys.argv[3] if len(sys.argv) > 3
           else os.path.join(os.path.dirname(base) or ".", "shapes.json"))
    json.dump(dict(rows=rows, worst=worst, block=rows[0].get("block")),
              open(out, "w"), indent=1)
    print(f"-> {out}")
