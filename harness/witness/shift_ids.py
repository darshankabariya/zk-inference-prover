"""B10 step 4: the recorded ids re-windowed one position back.

Why this exists.  The head's rows are a causal language model's rows, so the row
at position `p` produces the logits that predict position `p + 1`.  The recorded
rectangle is the 512 decoded tokens themselves, positions `[P, P + 512)`, so its
rows emit the window shifted by one and the last row's token is not in the ids at
all -- `extract.py` counts that as `emits_next_token` 511 of 511.  The argmax
statement wants the other alignment: the row that emits a token should *be* that
token's row, 512 of 512.

That is not something a different `gen_ids.py --offset` can fix.  The off-by-one
is structural -- any freshly generated continuation reproduces it exactly -- so
the fix is to move the 512-row window one position earlier over the *same*
recorded tokens, which is what this script writes.

The construction, and why it is the only one.  The extractor prefills in whole
`--chunk` blocks from index 0 and keeps the last one, so the fed array must be
`P + 512` long for the rectangle to be 512 rows on a `P`-token prefix.  The
rectangle has to end at recorded index `P + 510`, so the array has to *start* one
token before the recorded array:

    shifted = [lead] || recorded[0 : P + 511]

    rectangle = shifted[-512:] = recorded[P - 1 : P + 511]
              = the last prompt token, then 511 of the 512 decoded tokens

and those 512 rows emit exactly the 512 decoded tokens `recorded[P : P + 512)`.
The prefix stays `P` -- C is still a power of two, which the master-table sizing
and the no-wrap window both depend on -- and S is still 512.

`lead` is the corpus token immediately before the prompt, so the shift is a real
one-token move of the prompt window and nothing is invented:

  * C=4K takes its prompt from corpus offset 12288, so `lead` is corpus[12287].
  * C=16K takes its prompt from corpus offset 0.  There is no corpus token before
    it, so `lead` is BOS repeated -- the corpus already begins with BOS, and the
    duplicate sits 16,384 positions away from the rectangle, at the one end of
    the context the alignment check does not read.  It is recorded rather than
    hidden: the note beside each output says which of the two cases applied.

The continuation is never regenerated; its last token simply falls outside the
shifted rectangle.

    uv run python b10/gpu/shift_ids.py --ids w8b/ids_c4096.npy --prefix 4096 \
        --offset 12288 --out w8b/ids_c4096_s1.npy
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "b1"))

from common import tokens  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True, help="the recorded prompt || continuation (.npy)")
    p.add_argument("--prefix", type=int, required=True, help="the committed KV prefix, P")
    p.add_argument("--offset", type=int, required=True,
                   help="the prompt's first corpus token, from the ids' .json note")
    p.add_argument("--new", type=int, default=512, help="decoded tokens in the rectangle")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ids = np.load(a.ids).astype(np.int64)
    P, S = a.prefix, a.new
    assert len(ids) == P + S, f"{a.ids} has {len(ids)} ids, expected {P + S}"

    if a.offset > 0:
        ev, _ = tokens()
        ev = ev.numpy().astype(np.int64)
        assert np.array_equal(ids[:P], ev[a.offset:a.offset + P]), \
            "the recorded prompt is not the corpus at the offset its note claims"
        lead, lead_kind = int(ev[a.offset - 1]), "corpus"
    else:
        # The prompt starts at corpus token 0; there is nothing before it to take.
        lead, lead_kind = int(ids[0]), "repeated-bos"

    shifted = np.concatenate([np.array([lead], dtype=np.int64), ids[:P + S - 1]])

    # The three properties the rest of the step relies on, asserted rather than
    # assumed -- getting any of them wrong is silent downstream.
    assert len(shifted) == P + S, f"shifted is {len(shifted)}, expected {P + S}"
    assert np.array_equal(shifted[-S:], ids[P - 1:P + S - 1]), \
        "the shifted rectangle is not the recorded one moved back a position"
    assert np.array_equal(shifted[1:], ids[:P + S - 1]), \
        "the shifted array is not the recorded ids behind one lead token"

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.save(a.out, shifted)
    note = dict(source=os.path.abspath(a.ids), prefix=P, new=S, offset=a.offset,
                shifted_by=1, total=len(shifted), lead_token=lead, lead_kind=lead_kind,
                rectangle_recorded_positions=[P - 1, P + S - 1],
                emits_recorded_positions=[P, P + S],
                target_tokens=[int(x) for x in ids[P:P + S]][:8])
    json.dump(note, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"wrote {a.out}: {len(shifted)} ids, lead {lead} ({lead_kind}); "
          f"rectangle = recorded [{P - 1}, {P + S - 1}), emitting recorded [{P}, {P + S})")
