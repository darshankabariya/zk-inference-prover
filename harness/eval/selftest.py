"""B8 step 1: does B1's scheme run at 8B, and is the forward pass still the model?

  1. our fp32 forward vs the HuggingFace fp32 forward on 128 tokens
     (b1/selftest.py test 3, on Llama-3.1-8B-Instruct on the GPU);
  2. the integer paths under `b6-design-p2o`: max |accumulator| per matmul kind
     against the M31 signed range (H2 at K = 14336), and against 2^24, the
     line above which an fp32 copy of the accumulator stops being exact;
  3. the exp-table precondition (B6 section 6): per layer, the spread of
     `frac_per_head` = -log2(s_q s_k a) over heads -- the B6.5 AIR commits one
     master exp table per layer, which held on 4 of Qwen-0.5B's 24 layers;
  4. how often the calibrated attention scales clip on the eval slice;
  5. batched vs token-at-a-time evaluation under `b6-design-p2o` (b1 section 5).

    uv run python -m harness.eval.selftest   # needs a GPU and the 8B model
"""
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from harness.witness.common import MODEL, OUT, SPECS, chunks, setup, tokens
from harness.witness.model import Cache, Model
from harness.witness.ppl import attention_scales

P = (1 << 31) - 1
M31_MAX = (P - 1) // 2
N_TOK = 128
KINDS = ["q", "k", "v", "o", "gate", "up", "down", "lm_head"]
results = {}


def test_vs_hf(ids):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    ours_m = Model(MODEL, SPECS["fp32"])
    ours, _ = ours_m.forward(ids)
    ours = ours.float().cpu()
    results["model"] = dict(path=MODEL, layers=ours_m.L, d=ours_m.d, nh=ours_m.nh, nkv=ours_m.nkv,
                            hd=ours_m.hd, vocab=ours.shape[1], tied=ours_m.tied,
                            q_fold=ours_m.q_fold, attn_scale=ours_m.attn_scale,
                            rope=dict(ours_m.cfg.rope_parameters))
    del ours_m
    torch.cuda.empty_cache()
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda").eval()
    theirs = hf(ids.unsqueeze(0).cuda()).logits[0].float().cpu()
    del hf
    torch.cuda.empty_cache()
    d = (ours - theirs).abs()
    rel = d.max().item() / (theirs.max() - theirs.min()).item()
    agree = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    print(f"1. vs HuggingFace (fp32, {len(ids)} tokens): max|dlogit| {d.max().item():.3e}  "
          f"({rel:.1e} of logit range)  mean {d.mean().item():.3e}  argmax agree {agree * 100:.1f}%  "
          f"({time.time() - t0:.0f}s)")
    results["hf_max_abs_logit_diff"] = d.max().item()
    results["hf_mean_abs_logit_diff"] = d.mean().item()
    results["hf_argmax_agreement"] = agree
    return d.max().item() < 1e-2 and agree > 0.99


def test_accumulators(ids, spec_name="b6-design-p2o"):
    spec = SPECS[spec_name]
    m = Model(MODEL, spec, attn_scales=attention_scales(spec), track_acc=True)
    m.clamped = {}
    m.forward(ids)
    by_kind = {}
    for name, v in m.acc_max.items():
        kind = name.split(".", 1)[-1]
        by_kind[kind] = max(by_kind.get(kind, 0.0), v)
    print(f"2. max |accumulator| under `{spec_name}` over {len(ids)} tokens:")
    ok = True
    for k in KINDS + ["qk", "av"]:
        if k not in by_kind:
            continue
        v = by_kind[k]
        n_in = {"q": m.d, "k": m.d, "v": m.d, "o": m.d, "gate": m.d, "up": m.d,
                "down": m.cfg.intermediate_size, "lm_head": m.d, "qk": m.hd, "av": len(ids)}[k]
        ok &= v <= M31_MAX
        print(f"   {k:<8} K={n_in:<6} max {int(v):>12,}  x{M31_MAX / v:8.0f} to M31  "
              f"{'> 2^24' if v > (1 << 24) else '< 2^24'}")
    results["acc_max"] = by_kind
    results["clamped"] = {k: dict(clipped=v[0], total=v[1], frac=v[0] / v[1])
                          for k, v in m.clamped.items()}
    print("4. calibrated attention scales clipping on the eval slice: " +
          ", ".join(f"{k} {v[0]}/{v[1]} ({v[0] / v[1] * 100:.4f}%)" for k, v in m.clamped.items()))
    del m
    torch.cuda.empty_cache()
    return ok


def test_exp_table(spec_name="b6-design-p2o"):
    spec = SPECS[spec_name]
    sc = attention_scales(spec)
    layers = sorted({int(k[1:].split(".")[0]) for k in sc})
    spreads, fracs = {}, {}
    for li in layers:
        s_q, s_k = sc[f"L{li}.q"].flatten(), sc[f"L{li}.k"].flatten()
        # the attention scale is folded into W_q at hd = 128 (model.py), so a = 1
        frac = (-torch.log2(s_q * s_k)).round().to(torch.int64).tolist()
        fracs[li] = frac
        spreads[li] = max(frac) - min(frac)
    hist = {}
    for s in spreads.values():
        hist[s] = hist.get(s, 0) + 1
    one = sum(1 for s in spreads.values() if s == 0)
    print(f"3. exp-table exponent per layer (`frac_per_head`, B6 section 6): "
          f"{one}/{len(layers)} layers with one master exponent (spread 0); "
          f"spread histogram {dict(sorted(hist.items()))}; "
          f"max spread {max(spreads.values())}")
    print("   per layer: " + " ".join(f"L{li}:{spreads[li]}" for li in layers))
    results["frac_spread"] = spreads
    results["frac_per_head"] = fracs
    return True


def test_order(ids, spec_name="b6-design-p2o"):
    spec = SPECS[spec_name]
    m = Model(MODEL, spec, attn_scales=attention_scales(spec))
    batched, _ = m.forward(ids)
    cache = Cache(m.L)
    seq = []
    for t in ids:
        lg, cache = m.forward(t.view(1), cache)
        seq.append(lg)
    seq = torch.cat(seq)
    d = (batched - seq).abs()
    agree = (batched.argmax(-1) == seq.argmax(-1)).float().mean().item()
    ident = bool((batched == seq).all().item())
    print(f"5. batched vs token-at-a-time under `{spec_name}` on {len(ids)} tokens: "
          f"max|dlogit| {d.max().item():.3e}, argmax agree {agree * 100:.1f}%, bit-identical: {ident}")
    results["order"] = dict(max_abs=d.max().item(), argmax_agreement=agree, identical=ident)
    del m
    torch.cuda.empty_cache()
    return agree > 0.95


def test_packed(spec_name="b6-design-p2o", n_prompts=6, n_new=48, batch=4):
    """Packed decode (b8/tasks.py) against one-prompt-at-a-time decode: the same
    prompts, the same tokens, bit for bit -- the packing is a scheduling change
    only (b1 section 5's question, asked of the decode harness)."""
    from .tasks import STOP, chat_ids, generate, generate_packed, load_ifeval
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = load_ifeval()[:n_prompts]
    prompts = [(d["id"], torch.tensor(chat_ids(tok, d["prompt"]))) for d in docs]
    spec = SPECS[spec_name]
    m = Model(MODEL, spec, attn_scales=attention_scales(spec) if spec.int_softmax else None)
    t0 = time.time()
    single = {pid: generate(m, ids, n_new) for pid, ids in prompts}
    t_single = time.time() - t0
    packed = {}
    t0 = time.time()
    generate_packed(m, prompts, n_new, batch, lambda pid, out, fin: packed.__setitem__(pid, (out, fin)))
    t_packed = time.time() - t0
    same = all(single[p] == packed[p] for p in single)
    n_tok = sum(len(v[0]) for v in single.values())
    print(f"6. packed decode (batch {batch}) vs one at a time, {n_prompts} prompts x {n_new} tokens "
          f"under `{spec_name}`: identical token streams: {same}; "
          f"{n_tok / t_single:.1f} -> {n_tok / t_packed:.1f} tok/s")
    results["packed"] = dict(identical=same, tok_s_single=n_tok / t_single,
                             tok_s_packed=n_tok / t_packed, batch=batch)
    del m
    torch.cuda.empty_cache()
    return same


if __name__ == "__main__":
    setup()
    ev, _ = tokens()
    ids = chunks(ev)[0][:N_TOK]
    t0 = time.time()
    if "--packed" in sys.argv:
        passed = [test_packed()]
    else:
        passed = [test_vs_hf(ids), test_accumulators(chunks(ev)[0]), test_exp_table(),
                  test_order(chunks(ev)[1][:64]), test_packed()]
    results["passed"] = all(passed)
    results["seconds"] = round(time.time() - t0, 1)
    path = os.path.join(OUT, "selftest.json")
    if "--packed" in sys.argv and os.path.exists(path):   # merge into the full run's record
        prev = json.load(open(path))
        prev["packed"] = results["packed"]
        results = prev
    json.dump(results, open(path, "w"), indent=1)
    print(f"{'PASS' if all(passed) else 'FAIL'} ({results['seconds']}s) -> out/selftest.json")
