"""E5c: is B1's Gemma-4 forward pass the same model transformers implements?

The 31B checkpoint is 62 GB and this stream is CPU-only, so faithfulness is
established the way a compiler is: **component by component, against
`transformers`' own `models/gemma4` modules**, on synthetic inputs.  Six things
separate Gemma-4 from the Gemma-3 path B1 already had, and each gets its own row:

  1. the RoPE tables -- `default` on the local layers, `proportional` with
     `partial_rotary_factor` 0.25 (64 rotated pairs, 192 NoPE) on the global ones;
  2. the per-layer-type head geometry, checked on the *real* 31B config against a
     meta-device instantiation of the real 60-layer tower;
  3. the `attention_k_eq_v` value path -- V is `k_proj(x)` before K's norm, through
     a scale-free RMSNorm, never rotated;
  4. the `w` (not `1 + w`) norm gain, and the scale-free V-norm;
  5. the per-layer `layer_scalar`;
  6. the `tanh` logit softcap at 30.0.

Rows 3-6 need weights, so a structurally faithful *small* Gemma-4 is built with
random weights, saved as bf16 (the dtype a real checkpoint is stored in, so our
bf16 load and HF's fp32 load see the same numbers), and driven both ways.  The
whole-model row goes through `Gemma4ForConditionalGeneration` -- the multimodal
wrapper the 31B checkpoint actually is -- so the text-tower extraction path is
covered too.  Two controls (the sliding-window boundary, and the two constants)
show the checks would fail if the implementation were wrong.

    uv run python b1/gemma4_check.py

This is *not* the 31B gate.  A max-|dlogit| comparison against HuggingFace on the
real `google/gemma-4-31B` weights is still outstanding and must clear on a box
before any E5c quality number is trusted -- see briefs/e5c-gemma4.md.
"""
import gc
import json
import os
import shutil
import tempfile

import torch

from .common import SPECS
from .model import Model, proportional_inv_freq

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("ZK_THREADS", 4)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ZK_OUT", os.path.join(HERE, "out"))
# the published 31B config, fetched by E5a; `google/gemma-4-31B` is apache-2.0
# and ungated, and only the config is needed here -- never the weights
CFG31 = os.environ.get("ZK_GEMMA4_CONFIG", os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--google--gemma-4-31B", "blobs",
    "b469f1aca87e14dab2fcf32d071d3c58043357b7"))

rows, results = [], {}


def row(name, delta, note=""):
    rows.append((name, delta, note))
    results[name] = delta
    print(f"   {name:<44} {delta:>12.3e}  {note}")


def dmax(a, b):
    return (a.float() - b.float()).abs().max().item()


# --- the real 31B config: tables and geometry, no weights --------------------

def real_config():
    from transformers.models.gemma4 import Gemma4TextConfig
    return Gemma4TextConfig(**json.load(open(CFG31))["text_config"])


def bare_model(cfg):
    """A `Model` with only the fields `geometry_of` needs -- the geometry is read
    off the config, so it can be checked without loading 62 GB of weights."""
    m = Model.__new__(Model)
    m.spec, m.device, m.nh = SPECS["fp32"], "cpu", cfg.num_attention_heads
    m.hd = None
    return m


def test_rope_tables(cfg):
    """Our fp64 RoPE frequencies against `Gemma4TextRotaryEmbedding`'s own."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    print("1. RoPE tables on the real 31B config, vs Gemma4TextRotaryEmbedding:")
    m = bare_model(cfg)
    rot = Gemma4TextRotaryEmbedding(cfg)
    for t in ("sliding_attention", "full_attention"):
        hd = int(cfg.per_layer_config[t].head_dim)
        ours = m.inv_freq_of(dict(cfg.rope_parameters[t]), hd)
        theirs = getattr(rot, f"{t}_inv_freq").double()
        row(f"inv_freq {t} (hd {hd})", (ours - theirs).abs().max().item(),
            f"{int((theirs == 0).sum())}/{len(theirs)} NoPE")
        # The cos/sin a layer actually consumes.  B1's contract builds the table in
        # fp64 and rounds once; HF forms the angle in fp32, so at large positions
        # *their* table is the approximate one -- 4096 rad at fp32 is ~2.4e-4 of
        # angle error on its own.  Both position ranges are reported so the reader
        # can see which is which.
        for tag, pos in (("<=128", torch.arange(128)), ("<=4096", torch.arange(0, 4096, 37))):
            ang = torch.cat([pos.double().unsqueeze(-1) * ours.unsqueeze(0)] * 2, -1)
            c2, s2 = rot(torch.zeros(1, 1, dtype=torch.float32), pos.unsqueeze(0), t)
            row(f"cos/sin {t} (pos {tag})",
                max(dmax(ang.cos().float(), c2[0]), dmax(ang.sin().float(), s2[0])))
    # the NoPE half is position-free by construction: check it, do not assume it
    g = m.inv_freq_of(dict(cfg.rope_parameters["full_attention"]), 512)
    pairs = int(0.25 * 512 // 2)
    ok = bool((g[pairs:] == 0).all()) and bool((g[:pairs] > 0).all())
    print(f"   partial rotary: {pairs} rotated pairs, {len(g) - pairs} NoPE, "
          f"split exact: {ok}")
    results["partial_rotary_split"] = ok
    return ok


def test_geometry(cfg):
    """Per-layer-type head geometry, all 60 layers, against the real tower built
    on the meta device (structure without a single allocated weight)."""
    from transformers.models.gemma4 import Gemma4ForCausalLM
    print("2. head geometry on the real 31B config, vs a meta-device Gemma4ForCausalLM:")
    m = bare_model(cfg)
    geoms = {t: m.geometry_of(cfg.per_layer_config[t]) for t in set(cfg.layer_types)}
    with torch.device("meta"):
        hf = Gemma4ForCausalLM(cfg)
    bad, seen = [], {}
    for i, layer in enumerate(hf.model.layers):
        a, t = layer.self_attn, cfg.layer_types[i]
        g = geoms[t]
        want = (a.head_dim, a.config.num_attention_heads // a.num_key_value_groups,
                a.num_key_value_groups)
        got = (g["hd"], g["nkv"], g["rep"])
        has_v = a.v_proj is not None
        widths = (a.q_proj.out_features, a.k_proj.out_features)
        ours_widths = (m.nh * g["hd"], g["nkv"] * g["hd"])
        if got != want or widths != ours_widths or has_v != (t == "sliding_attention"):
            bad.append((i, t, got, want, widths, ours_widths, has_v))
        seen[t] = (got, widths, has_v)
    del hf
    gc.collect()
    for t, (g, w, has_v) in sorted(seen.items()):
        print(f"   {t:<18} hd {g[0]:>4}  kv heads {g[1]:>3}  rep {g[2]:>2}  "
              f"q_proj out {w[0]:>6}  k_proj out {w[1]:>5}  v_proj: {has_v}")
    print(f"   all {len(cfg.layer_types)} layers agree: {not bad}")
    results["geometry_layers_disagreeing"] = len(bad)
    return not bad


# --- a structurally faithful small Gemma-4, with weights ---------------------

SMALL = dict(
    vocab_size=1024, hidden_size=384, intermediate_size=768, num_hidden_layers=12,
    num_attention_heads=8, num_key_value_heads=4, head_dim=32,
    global_head_dim=64, num_global_key_value_heads=2,
    hidden_activation="gelu_pytorch_tanh", max_position_embeddings=512,
    rms_norm_eps=1e-6, tie_word_embeddings=True, sliding_window=16,
    final_logit_softcapping=30.0, attention_k_eq_v=True,
    hidden_size_per_layer_input=0, num_kv_shared_layers=0, enable_moe_block=False,
    use_double_wide_mlp=False, use_bidirectional_attention="vision",
    pad_token_id=0, eos_token_id=1, bos_token_id=2,
)


def build_small(path):
    """A random Gemma-4 with the 31B's *structure*, saved bf16 as a checkpoint is.

    Every norm gain and every `layer_scalar` is randomized away from 1: with the
    initialized values (all ones) the `w` vs `1 + w` question and the layer scalar
    would both be invisible.
    """
    from transformers.models.gemma4 import (Gemma4Config, Gemma4TextConfig,
                                            Gemma4ForConditionalGeneration)
    tcfg = Gemma4TextConfig(**SMALL)
    # the multimodal token ids have to be real in-vocab ids: HF's wrapper builds
    # its placeholder mask from them even on a text-only call.  Nothing in the
    # test ever emits one (`ids` are drawn below vocab_size - 8).
    cfg = Gemma4Config(text_config=tcfg, vision_config=None, audio_config=None,
                       boi_token_id=1016, eoi_token_id=1017, image_token_id=1018,
                       video_token_id=1019, boa_token_id=1020, eoa_token_index=1021,
                       audio_token_id=1022, tie_word_embeddings=True)
    hf = Gemma4ForConditionalGeneration(cfg)
    for n, p in hf.named_parameters():
        with torch.no_grad():
            if n.endswith("norm.weight") or "layernorm" in n:
                p.copy_(1.0 + 0.15 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
    for layer in hf.model.language_model.layers:
        layer.layer_scalar.copy_(0.8 + 0.4 * torch.rand(1))
    hf.eval().to(torch.bfloat16).save_pretrained(path)
    return tcfg


def hf_small(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def additive_mask(mask):
    """Our boolean (S, C) `hidden` mask as HF's additive (1, 1, S, C) float mask."""
    return torch.where(mask, torch.tensor(float("-inf")), torch.tensor(0.0))[None, None]


def test_components(path, ids):
    """Stage by stage through one local and one global layer.

    The reference for every stage is a `transformers` object -- the layer's own
    `q_proj`/`q_norm`/`k_proj`/`v_norm`/`o_proj` submodules, `apply_rotary_pos_emb`,
    `eager_attention_forward` and finally the whole `Gemma4TextDecoderLayer`.
    """
    from transformers.models.gemma4.modeling_gemma4 import (apply_rotary_pos_emb,
                                                            eager_attention_forward)
    print("3. component by component, small model (local layer 0, global layer 5):")
    hf = hf_small(path)
    lm = hf.model.language_model
    m = Model(path, SPECS["fp32"])
    S = len(ids)
    pos = torch.arange(S)
    x = torch.randn(S, m.d) * 3.0            # the residual stream is large in Gemma
    causal = pos.unsqueeze(0) > pos.unsqueeze(1)

    for li, tag in ((0, "local"), (5, "global")):
        t = m.layer_types[li]
        assert t == ("sliding_attention" if tag == "local" else "full_attention")
        m.use_geometry(t)
        l, a = m.layers[li], lm.layers[li].self_attn
        hd, nkv = m.hd, m.nkv
        mask = causal if t == "full_attention" else (
            causal | (pos.unsqueeze(0) <= pos.unsqueeze(1) - m.sliding_window))
        cos, sin = m.rope(pos, t)
        cos2, sin2 = lm.rotary_emb(x.unsqueeze(0), pos.unsqueeze(0), t)

        # -- norms, on their own
        row(f"[{tag}] RMSNorm gain w (norm1)",
            dmax(m.rmsnorm(x, l["norm1"], "t"), lm.layers[li].input_layernorm(x)))

        # -- Q: project, per-head norm over head_dim, rotate
        qh = a.q_proj(x).view(S, m.nh, hd)
        row(f"[{tag}] q_proj", dmax(l["q"](x, m).view(S, m.nh, hd), qh))
        qn = a.q_norm(qh)
        row(f"[{tag}] q_norm (over head_dim)",
            dmax(m.rmsnorm(l["q"](x, m).view(S, m.nh, hd), l["qn"], "t", m.inv_hd), qn))
        qr = apply_rotary_pos_emb(qn.unsqueeze(0), cos2, sin2, unsqueeze_dim=2)[0]
        ours_q = m.rotate(m.rmsnorm(l["q"](x, m).view(S, m.nh, hd), l["qn"], "t", m.inv_hd),
                          cos.unsqueeze(1), sin.unsqueeze(1))
        row(f"[{tag}] q after RoPE", dmax(ours_q, qr))

        # -- K and V.  On a global layer there is no v_proj: V is this same
        #    projection, taken *before* k_norm, through the scale-free v_norm.
        kh = a.k_proj(x).view(S, nkv, hd)
        vh = kh if a.v_proj is None else a.v_proj(x).view(S, nkv, hd)
        ours_kv = l["k"](x, m).view(S, nkv, hd)
        row(f"[{tag}] k_proj", dmax(ours_kv, kh))
        row(f"[{tag}] v source is k_proj: {a.v_proj is None}",
            dmax(ours_kv if l["v"] is None else l["v"](x, m).view(S, nkv, hd), vh))
        ours_k = m.rotate(m.rmsnorm(ours_kv, l["kn"], "t", m.inv_hd),
                          cos.unsqueeze(1), sin.unsqueeze(1))
        kr = apply_rotary_pos_emb(a.k_norm(kh).unsqueeze(0), cos2, sin2, unsqueeze_dim=2)[0]
        row(f"[{tag}] k after norm + RoPE", dmax(ours_k, kr))
        ours_v = m.rmsnorm(ours_kv if l["v"] is None else l["v"](x, m).view(S, nkv, hd),
                           l["vn"], "t", m.inv_hd)
        row(f"[{tag}] v_norm (scale-free, unrotated)", dmax(ours_v, a.v_norm(vh)))
        # control: V is *not* rotated.  Rotating it has to move the answer.
        vrot = m.rotate(ours_v, cos.unsqueeze(1), sin.unsqueeze(1))
        results[f"{tag}_control_v_rotated"] = dmax(vrot, a.v_norm(vh))
        print(f"   [{tag}] control: rotating V instead would cost "
              f"{results[f'{tag}_control_v_rotated']:.3e}")

        # -- the attention block, against transformers' own eager kernel at
        #    scaling=1.0 (Gemma4TextAttention.scaling is literally 1.0)
        assert m.attn_scale == 1.0 and a.scaling == 1.0
        ao, _ = eager_attention_forward(
            a, qr.transpose(0, 1).unsqueeze(0), kr.transpose(0, 1).unsqueeze(0),
            a.v_norm(vh).transpose(0, 1).unsqueeze(0), additive_mask(mask), scaling=1.0)
        ours_a, _ = m.attention(ours_q, ours_k, ours_v, mask, li)
        row(f"[{tag}] attention out (pre o_proj)",
            dmax(ours_a, ao.reshape(S, m.nh * hd)))
        row(f"[{tag}] o_proj", dmax(l["o"](ours_a, m), a.o_proj(ao.reshape(S, m.nh * hd))))

        # -- the whole decoder layer, incl. the two extra norms and layer_scalar
        theirs = lm.layers[li](x.unsqueeze(0), position_embeddings=(cos2, sin2),
                               attention_mask=additive_mask(mask),
                               shared_kv_states={}, use_cache=False)
        theirs = theirs[0] if isinstance(theirs, tuple) else theirs
        from .model import Cache
        m.use_geometry(t)
        ours = m.gemma4_layer(li, l, x, Cache(m.L), (cos, sin), mask)
        row(f"[{tag}] decoder layer (4 norms + layer_scalar)", dmax(ours, theirs[0]))
        # control: the layer scalar is not 1
        sc = float(l["scalar"].item())
        results[f"{tag}_layer_scalar"] = sc
        print(f"   [{tag}] control: layer_scalar = {sc:.6f}; dropping it would cost "
              f"{dmax(ours / sc, theirs[0]):.3e}")

    # -- the logit softcap, on its own
    z = torch.randn(64, 128) * 40.0
    cap = m.softcap
    ours_cap = m.store(m.unary("t", m.store(z / cap), torch.tanh) * cap)
    row("logit softcap (tanh, 30.0)", dmax(ours_cap, torch.tanh(z / cap) * cap))
    del hf, m
    gc.collect()


def test_whole_model(path, ids):
    """The whole tower, through the multimodal wrapper the 31B checkpoint is."""
    print("4. whole model, through Gemma4ForConditionalGeneration:")
    m = Model(path, SPECS["fp32"])
    ours, _ = m.forward(ids)
    window = m.sliding_window
    del m
    gc.collect()
    hf = hf_small(path)
    theirs = hf(ids.unsqueeze(0)).logits[0].float()
    del hf
    gc.collect()
    d = (ours - theirs).abs().max().item()
    agree = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    row(f"logits, {len(ids)} tokens (window {window})", d,
        f"{d / theirs.abs().max().item():.1e} of range, argmax {agree * 100:.1f}%")
    results["whole_model_argmax_agreement"] = agree

    # control: the sliding-window boundary, the way E5a pinned Gemma-3's
    print(f"   control: the window boundary (an off-by-one either way)")
    ctl = {}
    for w in (window - 1, window, window + 1):
        m = Model(path, SPECS["fp32"])
        m.sliding_window = w
        o, _ = m.forward(ids)
        ctl[w] = (o - theirs).abs().max().item()
        print(f"     window {w:>3}: max|dlogit| {ctl[w]:.4e}")
        del m
        gc.collect()
    results["window_control"] = ctl
    return d < 5e-2 and agree == 1.0 and ctl[window] < min(ctl[window - 1], ctl[window + 1])


def variant(path, **over):
    """The small checkpoint with its `text_config` edited on disk.

    `Model` reads the config from the checkpoint, so a guard is only really
    exercised by a checkpoint that declares the feature.
    """
    d = tempfile.mkdtemp()
    shutil.rmtree(d)
    shutil.copytree(path, d)
    cj = os.path.join(d, "config.json")
    c = json.load(open(cj))
    c["text_config"].update(over)
    json.dump(c, open(cj, "w"))
    return d


def test_schedule(path, ids):
    """Batched vs token-at-a-time, and two sequences packed vs run apart.

    The head geometry is *mutable state* on the model (`use_geometry`), which is
    the one thing about this implementation that could go wrong re-entrantly, and
    B8's IFEval decodes many prompts in one packed cache.  Both are checked the
    way `selftest.py` test 6 checks the shared path.
    """
    from .model import Cache
    print("5. schedule invariance (the geometry is switched per layer, so: state):")
    ok = True
    for name in ("fp32", "dynamic-int8"):
        m = Model(path, SPECS[name])
        batched, _ = m.forward(ids)
        cache = Cache(m.L)
        seq = torch.cat([m.forward(ids[i:i + 1], cache)[0] for i in range(len(ids))])
        same = bool((batched == seq).all().item())
        d = (batched - seq).abs().max().item()
        agree = (batched.argmax(-1) == seq.argmax(-1)).float().mean().item()
        print(f"   {name:<11} batched vs one-at-a-time: bit-identical {same}, "
              f"max|d| {d:.2e}, argmax {agree * 100:.1f}%")
        results[f"seq_vs_batched_{name}"] = dict(bit_identical=same, max_abs=d, argmax=agree)

        # packed decode: two sequences in one cache, against each on its own
        a, b = ids[:40], ids[40:80]
        pos = torch.cat([torch.arange(len(a)), torch.arange(len(b))])
        sid = torch.cat([torch.zeros(len(a), dtype=torch.int64),
                         torch.ones(len(b), dtype=torch.int64)])
        packed, _ = m.forward(torch.cat([a, b]), Cache(m.L), pos, sid)
        alone = torch.cat([m.forward(a)[0], m.forward(b)[0]])
        psame = bool((packed == alone).all().item())
        print(f"   {name:<11} 2 sequences packed vs apart: bit-identical {psame}, "
              f"max|d| {(packed - alone).abs().max().item():.2e}")
        results[f"packed_{name}"] = psame
        # fp32 is allowed to differ by BLAS reduction order (selftest test 6's
        # finding on the shared path); under bf16 storage the schedule must not
        # matter at all, and that is the statement B8's packed decode rests on.
        ok &= (agree == 1.0 and d < 1e-2) if name == "fp32" else (same and psame)
        del m
        gc.collect()
    return ok


def test_ladder(path, ids):
    """Does the whole spec ladder execute on a Gemma-4 layer?

    A smoke run, not a result: the corpus is 96 random token ids on a randomly
    initialized model, so the NLLs mean nothing.  What it establishes is that the
    integer softmax, the calibrated-free K/V scales, the power-of-two attention
    scales and the int8 attention output all run over per-layer-type head dims,
    a `k_eq_v` value path and a `layer_scalar` without special casing.
    """
    print("6. the spec ladder executes on Gemma-4 (smoke only, not a result):")
    base = None
    for name in ("fp32", "dynamic-int8", "b10-final", "y-delay-vw12"):
        m = Model(path, SPECS[name])
        lo, _ = m.forward(ids)
        if base is None:
            base = lo
        nll = torch.nn.functional.cross_entropy(lo[:-1].float(), ids[1:]).item()
        agree = (lo.argmax(-1) == base.argmax(-1)).float().mean().item()
        print(f"   {name:<14} ran; mean NLL {nll:7.4f}, argmax vs fp32 {agree * 100:5.1f}%")
        results[f"ladder_{name}"] = nll
        del m
        gc.collect()
    return True


def refuses_all(path):
    """Anything this forward pass does not implement has to fail loudly."""
    print("7. refusals -- features in the Gemma-4 class but off in the 31B checkpoint:")
    ok = True
    # each case is the config edit that turns the feature on; the MoE one has to
    # set the expert dims too or HF's own constructor raises before our guard runs
    cases = [dict(hidden_size_per_layer_input=256), dict(num_kv_shared_layers=2),
             dict(use_double_wide_mlp=True),
             dict(enable_moe_block=True, num_experts=4, top_k_experts=2,
                  moe_intermediate_size=128),
             dict(use_bidirectional_attention="all"), dict(model_type="gemma5_text")]
    for over in cases:
        flag, val = list(over.items())[0]
        d = variant(path, **over)
        try:
            Model(d, SPECS["fp32"], device="cpu")
            got, why = False, "loaded"
        except NotImplementedError as e:
            got, why = True, str(e)[:46]
        except Exception as e:                      # a crash is not a refusal
            got, why = False, f"{type(e).__name__}: {str(e)[:34]}"
        finally:
            shutil.rmtree(d, ignore_errors=True)
            gc.collect()
        ok &= got
        print(f"   {flag}={val!r}: refused {got}  ({why})")
    results["refusals_all_raise"] = bool(ok)
    return bool(ok)


if __name__ == "__main__":
    cfg = real_config()
    passed = [test_rope_tables(cfg), test_geometry(cfg)]
    del cfg
    gc.collect()
    with tempfile.TemporaryDirectory() as d:
        build_small(d)
        ids = torch.randint(0, SMALL["vocab_size"] - 8, (96,))
        test_components(d, ids)
        passed.append(test_whole_model(d, ids))
        passed.append(test_schedule(d, ids))
        passed.append(test_ladder(d, ids))
        passed.append(refuses_all(d))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gemma4_check.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nall passed: {all(passed)}   -> {OUT}/gemma4_check.json")
