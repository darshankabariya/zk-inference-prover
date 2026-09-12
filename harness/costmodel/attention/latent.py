"""Multi-head latent attention, as the `params` block `exact_cells` counts.

    uv run python -m harness.costmodel.exact_cells --selftest

Three of the frontier's nine shapes -- DeepSeek-V3, Kimi K2 and Kimi K3 -- do not
project keys and values per head at all.  Both sides of the attention go through
a low rank: the queries through `q_lora_rank`, the keys and values through
`kv_lora_rank`, and the position is carried by a decoupled key of
`qk_rope_head_dim` that is computed once per key rather than once per head.  So
what a key costs is not `2 nh hd`, the layer's matmuls are five and not four, and
the score grid reads a width no field of the shape states.  This module is that
width, read off the five fields the configs publish
(`FIELDS`), and the two layouts a prover could build over it.

THE TWO FORMS.  A latent attention can be proved either way, and the difference
is a projection either way -- the choice is where the low-rank pair `kv_b` is
applied, not what the model computes:

  * `materialised` -- the keys and values are expanded per head before the
    attention, which is what a prover with no latent layout does: `kv_b` runs
    over the `S` keys of the block and writes `nh (qk_nope + v_head)` numbers a
    key, and the key tile family is the ordinary one at `hd_qk = qk_nope +
    qk_rope` and `hd_v = v_head`.
  * `absorbed` -- `kv_b` is folded into the query side and the output side
    instead (`W_UK` into the query, `W_UV` into the output projection), which is
    how the published implementations decode.  Then the cache IS the latent:
    `kv_lora_rank + qk_rope_head_dim` numbers a key for every head at once.  The
    score is a dot product over that width and the attention output accumulates
    in it, so `hd_qk = kv_lora + qk_rope` and `hd_v = kv_lora`, and two per-head
    matmuls appear that the materialised form does not have.

The weights are the same either way, to the parameter: `W_UK` and `W_UV`
together ARE `kv_b`, applied to `S` tokens on one side or `S` keys on the other.
What differs is the committed trace -- the cache, which the absorbed form holds
once instead of `nh` times, against two more matmuls a token and a wider score
grid -- and that is what the count is for.

WHAT IS NOT READ HERE.  No constant, no measured number and no fitted one: the
five fields of a config, the head count, and the widths that follow from them.
"""

# The five fields of a `config.json` that state a latent attention.  Every width
# below is one of them or a sum of them; nothing is solved and nothing is fitted.
FIELDS = ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
          "v_head_dim")

FORMS = ("absorbed", "materialised")

# `rope.rs`, `V_BITS`: the value side's quantize width, as for a dense layer.
V_BITS = 8


class Refused(Exception):
    """This config does not state a latent attention, with the reason."""


def dims(cfg, nh, form):
    """The widths one latent-attention layer works at.

    `cfg` carries `FIELDS`, `nh` is `num_attention_heads` and `form` is one of
    `FORMS` -- named by every caller, because which one a count uses is the whole
    question and a default would hide it.  `hd_qk` is the width the score grid's
    channel component runs over, `hd_v` the width its value side and the
    attention output run over, and
    `cache_shared` says whether one key tile holds one cache for all its heads
    (the absorbed latent) or one per head (the materialised expansion)."""
    if form not in FORMS:
        raise Refused(f"no such latent-attention form: {form!r}")
    missing = [f for f in FIELDS if cfg.get(f) is None]
    if missing:
        raise Refused(f"this config states no {', '.join(missing)}, and a latent "
                      f"attention's widths are those fields and nothing else")
    z = {f: int(cfg[f]) for f in FIELDS}
    qk = z["qk_nope_head_dim"] + z["qk_rope_head_dim"]
    latent = z["kv_lora_rank"] + z["qk_rope_head_dim"]
    out = dict(z, nh=int(nh), form=form, qk=qk, latent=latent)
    if form == "materialised":
        out.update(hd_qk=qk, hd_v=z["v_head_dim"], cache_shared=False,
                   # `boundary.rs`, `kv_sides`: what one key hands on.  A head
                   # holds its own K of `qk` and its own V of `v_head_dim`.
                   cache_per_key=nh * (qk + z["v_head_dim"]))
    else:
        out.update(hd_qk=latent, hd_v=z["kv_lora_rank"], cache_shared=True,
                   # one latent a key, whatever the head count
                   cache_per_key=latent)
    return out


def attn_params(z, d):
    """One layer's attention weights, which are the same in both forms.

    `W_UK` and `W_UV` are the two halves of `kv_b`: the materialised form applies
    them to the keys and the values, the absorbed form to the query and to the
    attention output, and either way the model holds them once."""
    q_out = z["nh"] * z["qk"]
    kv_b = z["kv_lora_rank"] * z["nh"] * (z["qk_nope_head_dim"] + z["v_head_dim"])
    return (d * z["q_lora_rank"] + z["q_lora_rank"] * q_out + d * z["latent"]
            + kv_b + z["nh"] * z["v_head_dim"] * d)


def attn_accs(z, d):
    """The accumulators one token's attention projections produce, which the two
    forms do NOT share: the absorbed form pays `W_UK` at `kv_lora_rank` a head
    where the materialised form pays it at `qk_nope_head_dim` a key."""
    nh = z["nh"]
    base = z["q_lora_rank"] + nh * z["qk"] + z["latent"] + d
    if z["form"] == "materialised":
        return base + nh * (z["qk_nope_head_dim"] + z["v_head_dim"])
    return base + nh * (z["kv_lora_rank"] + z["v_head_dim"])


def layer(z, d, tokens, rope=True):
    """The matmuls, the requantize seams, the quantize sides and the extra norms
    one latent-attention layer builds, for `segment_params` to put in a block.

    `rope=False` is a config that says its latent attention applies no rotation
    (`mla_use_nope`): the widths do not move -- the decoupled key is still there
    and still cached -- but the wiring family builds no rotation.
    """
    nh, q_lora, kv_lora = z["nh"], z["q_lora_rank"], z["kv_lora_rank"]
    nope, v_head = z["qk_nope_head_dim"], z["v_head_dim"]
    # the two low-rank reads of the normed residual share one committed input, so
    # they are one Freivalds group, the one a dense layer calls `qkv`
    mats = [dict(name="q_a", n_in=d, n_out=q_lora, tokens=tokens, variant="chain",
                 group="qkv"),
            dict(name="kv_a", n_in=d, n_out=z["latent"], tokens=tokens,
                 variant="chain", group="qkv"),
            dict(name="q_b", n_in=q_lora, n_out=nh * z["qk"], tokens=tokens,
                 variant="chain", group="q_b")]
    # a seam is (name, the width it requantizes, the matmuls it feeds)
    seams = [("qkv", d, ("q_a", "kv_a")), ("q_lat", q_lora, ("q_b",))]
    if z["form"] == "materialised":
        mats.append(dict(name="kv_b", n_in=kv_lora, n_out=nh * (nope + v_head),
                         tokens=tokens, variant="chain", group="kv_b"))
        seams.append(("kv_lat", kv_lora, ("kv_b",)))
        quant = [dict(name="q", n_out=nh * z["qk"], tokens=tokens, hd=z["qk"],
                      bits="q_bits"),
                 dict(name="k", n_out=nh * z["qk"], tokens=tokens, hd=z["qk"],
                      bits="k_bits"),
                 dict(name="v", n_out=nh * v_head, tokens=tokens, hd=v_head,
                      bits=V_BITS)]
    else:
        # `W_UK` on the query side and `W_UV` on the output side: one matmul per
        # head, block diagonal, so each is its own layout and its own check
        mats.append(dict(name="q_absorb", n_in=nope, n_out=kv_lora, tokens=tokens,
                         variant="chain", group="q_absorb", repeat=nh))
        mats.append(dict(name="v_absorb", n_in=kv_lora, n_out=v_head, tokens=tokens,
                         variant="chain", group="v_absorb", repeat=nh))
        seams += [("q_nope", nh * nope, ("q_absorb",)),
                  ("v_out", nh * v_head, ("o",))]
        # the query is quantized in the latent's own width; the cache is
        # quantized once and is both the key side and the value side of the grid
        quant = [dict(name="q", n_out=nh * z["latent"], tokens=tokens,
                      hd=z["latent"], bits="q_bits"),
                 dict(name="k", n_out=z["latent"], tokens=tokens, hd=z["latent"],
                      bits="k_bits")]
    mats.append(dict(name="o", n_in=nh * v_head, n_out=d, tokens=tokens,
                     variant="chain"))
    return dict(mats=mats, seams=seams, quant_sides=quant,
                # `q_a_layernorm` and `kv_a_layernorm`: the plain norm the output
                # head's final norm already builds, at the two latents' widths
                norms=[q_lora, kv_lora],
                # the rotation is over the decoupled key alone, and that key is
                # one per key and not one per head
                rope=None if not rope else dict(nkv=1, hd=z["qk_rope_head_dim"]))
