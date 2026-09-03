"""B10 step 4: the extraction array for the shifted token rectangle.

The bug.  The head's rows are a causal language model's rows, so the row at
position `p` emits the token at position `p + 1`.  With a prompt of `P` tokens
and a 512-token continuation, the extractor's last chunk is positions
`[P, P + 512)` and those rows emit `[P + 1, P + 513)` -- but the recorded array
stops at `P + 512`, so the last row's token is not in it and only 511 of the 512
rows can be checked at all.  That is the "511 of 511" the record carries, and no
choice of prompt fixes it: it is structural.

The fix, and which of the two directions it is.  "Shift by one" can mean move the
rows back a position, or move the token window forward one.  They are not
equivalent here, and only the second works:

  * Moving the *rows* back means feeding an array that starts one token earlier,
    so the last 512-chunk lands on `[P - 1, P + 511)`.  Keeping S at 512 and the
    prefix at exactly `P` then forces a lead token onto the front of the array --
    and that lead token is in the context of every row, which changes the logits.
    Measured: the head dump then reproduces only 433 of 512 rows of the model's
    own greedy continuation, against 512 of 512 for the same rows with their true
    contexts.  The construction defeats the check it exists to pass.
  * Moving the *token window* forward -- which is what `state-of-play.md` calls
    "a one-position shift of the token rectangle" -- leaves the rows, the prefix
    and the contexts exactly as they were.  Nothing is perturbed; the only thing
    that changes is which tokens the rows are checked against, and one extra
    greedy token has to exist so the last row has one.

So the continuation is decoded to `--new 513` and this writes the first
`P + 512` ids as the extractor's array.  The rectangle's rows stay at
`[P, P + 512)` on a prefix of exactly `P`, and the tokens they emit are
`[P + 1, P + 513)` -- all 512 of them recorded.

    uv run python -m harness.witness.shift_ids --ids ids_c4096_513.npy \
        --prefix 4096 --out w8b/ids_c4096_final_rect.npy
"""
import argparse
import json
import os

import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True,
                   help="prompt || continuation with ONE token more than the rectangle "
                        "(.npy, from gen_ids.py --new 513)")
    p.add_argument("--prefix", type=int, required=True, help="the committed KV prefix, P")
    p.add_argument("--new", type=int, default=512, help="rows in the rectangle")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ids = np.load(a.ids).astype(np.int64)
    P, S = a.prefix, a.new
    assert len(ids) == P + S + 1, \
        f"{a.ids} has {len(ids)} ids, expected {P + S + 1} (the rectangle plus one)"

    rect = ids[:P + S]

    # The three properties the rest of the step relies on, asserted rather than
    # assumed -- each of them is silent downstream if it is wrong.
    assert len(rect) == P + S, f"the extraction array is {len(rect)}, expected {P + S}"
    assert len(rect) % S == 0, \
        f"{len(rect)} is not a whole number of {S}-token chunks, so the last chunk is ragged"
    assert np.array_equal(rect[:P], ids[:P]), "the prefix is not the recorded prompt"

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.save(a.out, rect)
    note = dict(source=os.path.abspath(a.ids), prefix=P, rows=S, total=len(rect),
                rows_at_positions=[P, P + S],
                emits_positions=[P + 1, P + S + 1],
                emitted_tokens_head=[int(x) for x in ids[P + 1:P + 9]])
    json.dump(note, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"wrote {a.out}: {len(rect)} ids; rows at [{P}, {P + S}) on a {P}-token prefix, "
          f"emitting [{P + 1}, {P + S + 1})")
