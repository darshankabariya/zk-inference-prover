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

    uv run python -m harness.eval.selftest         # needs a GPU and the 8B model
    uv run python -m harness.eval.selftest --cpu   # no card, no weights, nothing downloaded

The five tests above are about the model, so they need the model and a card.
`--cpu` is the other half, and the one CI can run: the eval harness *around* the
model -- the task loaders, the decode `tasks.run` drives, the scoring, the
provenance block and the two writers -- against a model with no weights, the way
`eval/baseline` exercises its own harness against a mock backend.  Where a tiny
model's weights are already in the local cache it also runs one real forward
pass through the same task code; where they are not, it says so and does not
download them.  Nothing it writes leaves a temporary directory.
"""
import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from harness import provenance
from harness.eval import tasks as T
from harness.eval.e5t import report as E
from harness.eval.tasks import STOP
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


# --- the CPU path: the eval harness itself, with no card and no weights -----
#
# The five tests above are about the model, so they need the model and a card.
# What they do not cover is the harness around it -- the task loaders, the
# decode `tasks.run` drives, the scoring and the two writers -- and that is the
# part a change can break silently.  `--cpu` runs it against a model with no
# weights, the way `eval/baseline` runs its own harness against a mock backend,
# so a broken eval harness is found in CI and not on a host.
#
# Llama-3.1's vocabulary size, so `tasks.STOP` is inside it and the stop path
# the runner takes is the shipped one.
MOCK_VOCAB = 128264
# The tiny model the CPU path runs one real forward pass of, when its weights
# are already in the local cache.  Its own name, not `ZK_MODEL`: the CPU path
# must not pull an 8B model into memory because a host set that variable.
CPU_MODEL = os.environ.get("ZK_SELFTEST_CPU_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class MockTokenizer:
    """One id per whitespace-separated word; a one-character word is its code point.

    That rule pins the two things `tasks.run` reads out of a tokenizer without
    downloading one: every answer letter is a single token (`A` is 65), and so
    is the ` (` the assistant turn is opened with.
    """

    eos_token_id = min(STOP)

    class Encoding:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def ids(self, text):
        out = []
        for word in text.split():
            if len(word) == 1:
                out.append(ord(word))
                continue
            h = 0
            for ch in word:
                h = (h * 131 + ord(ch)) % (MOCK_VOCAB - 1024)
            out.append(1024 + h)
        return out

    def __call__(self, text, add_special_tokens=True):
        return self.Encoding(self.ids(text))

    def encode(self, text, add_special_tokens=True):
        return self.ids(text)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"w{i}" for i in ids if not (skip_special_tokens and i in STOP))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return " ".join(m["content"] for m in messages)


class MockModel:
    """A model-shaped object with no weights: the next token is a fixed function
    of the last token and its position.

    Enough of `Model`'s surface for the task runner -- the layer count, the
    embedding's device, and `forward(ids, cache, pos, seq)` returning the logits
    and the cache it appended to -- and the cache is the shipped `Cache`, so the
    packed decode's merge and keep are the real ones.  `variant` moves the
    logits so two specs disagree on some items and the paired error has
    something to be an error of.

    Its logits are not a prediction of anything, exactly as the baseline's mock
    backend's seconds are not a measurement of anything.  Nothing it produces is
    written outside the temporary directory this path runs in.
    """

    synthetic = True
    eos_token_id = min(STOP)

    def __init__(self, variant=0, n_layers=2, vocab=MOCK_VOCAB):
        self.L, self.vocab, self.variant = n_layers, vocab, variant
        self.emb = torch.zeros(1)          # `generate_packed` reads its device

    def next_id(self, token, pos):
        """The continuation, and a stop token about one step in seventeen."""
        if (token + pos) % 17 == 0:
            return self.eos_token_id
        nxt = (token * 31 + pos * 17 + 3) % self.vocab
        return nxt + 1 if nxt in STOP else nxt

    def forward(self, ids, cache=None, pos=None, seq=None):
        ids = ids.view(-1)
        cache = Cache(self.L) if cache is None else cache
        n = len(ids)
        if pos is None:                    # one sequence: the cache's own order
            pos = torch.arange(cache.n, cache.n + n)
        if seq is None:
            seq = torch.zeros(n, dtype=torch.long)
        for i in range(self.L):
            cache.append(i, torch.zeros(n, 1), torch.zeros(n, 1))
        cache.append_meta(pos, seq)
        logits = torch.zeros(n, self.vocab)
        letters = torch.arange(10)
        for r, (t, p) in enumerate(zip(ids.tolist(), pos.tolist())):
            # The multiple-choice reader takes the argmax over the ten letter
            # tokens only, so those ten carry a spread of their own.
            logits[r, ord("A"):ord("A") + 10] = ((letters * (t + p + 1 + self.variant)) % 13).float()
            logits[r, self.next_id(t, p)] = 100.0
        return logits, cache


def mock_documents(task, n, groups):
    """`n` multiple-choice items of the shape both loaders return."""
    docs = []
    for i in range(n):
        choices = [f"{task} item {i} choice {j}" for j in range(4)]
        docs.append(dict(id=f"{task}-{i}", text=T.mc_text(f"question {i}?", choices),
                         n=4, gold=i % 4, group=groups[i % len(groups)]))
    return docs


def write_ifeval(path, n):
    """IFEval's own on-disk format, so the shipped loader reads a real file."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(dict(key=1000 + i, prompt=f"write item {i} out in full",
                                    instruction_id_list=["punctuation:no_comma"],
                                    kwargs=[{}])) + "\n")


def write_parquets(data_dir, gpqa, mmlu):
    """The two multiple-choice sets in their own format, where pandas can write it.

    Returns whether the shipped loaders will be the ones that read them.  The
    format is parquet, which needs pandas and pyarrow; neither is a dependency
    of this package, so the usual answer here is no and the loaders are replaced
    by ones that return the same documents.
    """
    try:
        import pandas as pd
    except ImportError:
        return False
    rows = [dict({"Record ID": d["id"], "Question": f"question {i}?",
                  "Correct Answer": f"gpqa item {i} choice 0",
                  "High-level domain": d["group"]},
                 **{f"Incorrect Answer {j}": f"gpqa item {i} choice {j}" for j in (1, 2, 3)})
            for i, d in enumerate(gpqa)]
    mrows = [dict(question_id=d["id"], question=f"question {i}?", category=d["group"],
                  options=[f"mmlu item {i} choice {j}" for j in range(4)],
                  answer_index=d["gold"], answer=T.LETTERS[d["gold"]])
             for i, d in enumerate(mmlu)]
    os.makedirs(os.path.join(data_dir, "gpqa"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "mmlu"), exist_ok=True)
    try:
        pd.DataFrame(rows).to_parquet(T.GPQA)
        pd.DataFrame(mrows).to_parquet(T.MMLU)
    except (ImportError, ValueError):      # no parquet engine installed
        return False
    return True


def check(label, ok, bad):
    print(f"{'OK  ' if ok else 'FAIL'}  {label}")
    return bad + (not ok)


def cpu_loaders(data_dir, n):
    """The three loaders over a corpus of `n` items each, and what read it.

    IFEval is the shipped loader over a written file either way.  The two
    multiple-choice sets are read by the shipped loaders where a parquet writer
    is installed, and by stand-ins returning the same documents where it is not;
    the run says which of the two it did rather than reporting a check it
    skipped.
    """
    gpqa = mock_documents("gpqa", n, ["Physics", "Chemistry"])
    mmlu = mock_documents("mmlu", n, ["math", "law"])
    write_ifeval(T.IFEVAL, n)
    real = write_parquets(data_dir, gpqa, mmlu)
    if not real:
        T.LOADERS = {"gpqa": lambda: gpqa, "mmlu_pro": lambda: mmlu,
                     "ifeval": T.load_ifeval}
    return real


def cpu_tasks(tmp, n):
    """The loaders, the runner, the scoring and the two writers, on the mock model."""
    bad = 0
    data_dir = os.path.join(tmp, "data")
    os.makedirs(data_dir, exist_ok=True)
    T.DATA = data_dir
    T.GPQA = os.path.join(data_dir, "gpqa", "test.parquet")
    T.MMLU = os.path.join(data_dir, "mmlu", "test.parquet")
    T.IFEVAL = os.path.join(data_dir, "ifeval.jsonl")
    T.TASKS = os.path.join(tmp, "tasks")
    T.RESULTS = os.path.join(tmp, "tasks.json")
    T.MAX_NEW = 24                         # the 1,280-token cap, at this size
    real = cpu_loaders(data_dir, n)

    docs = {t: T.LOADERS[t]() for t in T.LOADERS}
    ok = True
    for task in ("gpqa", "mmlu_pro"):
        for d in docs[task]:
            ok &= (set(d) >= {"id", "text", "n", "gold", "group"}
                   and 0 <= d["gold"] < d["n"]
                   and f"({T.LETTERS[d['n'] - 1]})" in d["text"]
                   and f"({T.LETTERS[d['n']]})" not in d["text"])
        ok &= len(docs[task]) == n
    for d in docs["ifeval"]:
        ok &= set(d) >= {"id", "prompt", "instruction_id_list", "kwargs"}
    ok &= len(docs["ifeval"]) == n
    how = ("all three shipped" if real else
           "ifeval shipped; the two parquet ones stood in for, no pandas here")
    bad = check(f"1. the three task loaders return {n} documents each, every one with the "
                f"fields the runner reads ({how})", ok, bad)

    m = MockModel()
    prompts = [(f"p{i}", torch.tensor(T.chat_ids(MockTokenizer(), f"prompt number {i} please")))
               for i in range(6)]
    single = {pid: T.generate(m, ids, T.MAX_NEW) for pid, ids in prompts}
    packed = {}
    T.generate_packed(m, prompts, T.MAX_NEW, 3,
                      lambda pid, out, fin: packed.__setitem__(pid, (out, fin)))
    ok = sorted(packed) == sorted(single) and all(single[p] == packed[p] for p in single)
    bad = check(f"2. packed decode (batch 3) is one at a time, token for token, over "
                f"{len(prompts)} prompts", ok, bad)

    T.load_model = lambda spec: MockModel(variant=0 if spec == "fp32" else 1)
    sizes = lambda: {f: os.path.getsize(os.path.join(T.TASKS, f))
                     for f in sorted(os.listdir(T.TASKS))}
    with mock_tokenizer():
        for spec in ("fp32", "b10-final"):
            for task in ("gpqa", "mmlu_pro"):
                T.run(task, spec, 0, 1, None, 4)
        written = sizes()
        for spec in ("fp32", "b10-final"):     # a finished shard has nothing left to do
            T.run("gpqa", spec, 0, 1, None, 4)
    rows = [json.loads(l) for l in open(os.path.join(T.TASKS, "gpqa.fp32.0of1.jsonl"))]
    ok = (len(rows) == n and sorted(r["id"] for r in rows) == sorted(d["id"] for d in docs["gpqa"])
          and written == sizes())
    bad = check("3. the runner writes one row per item and resumes a finished shard "
                "without adding another", ok, bad)

    res = T.score()
    ok = True
    for spec in ("fp32", "b10-final"):
        for task in ("gpqa", "mmlu_pro"):
            r, rs = res[spec][task], E.rows(T.TASKS, task, spec)
            acc = sum(x["pred"] == x["gold"] for x in rs.values()) / len(rs)
            ok &= abs(r["acc"] - acc) < 1e-12 and r["n"] == n and r["complete"]
            ok &= set(r["by_group"]) == {d["group"] for d in docs[task]}
    for task in ("gpqa", "mmlu_pro"):
        a, b = res["fp32"][task], res["b10-final"][task]
        ok &= abs(b["paired_vs_fp32"]["delta"] - (b["acc"] - a["acc"])) < 1e-12
        ok &= 0.0 <= b["agree_fp32"] <= 1.0
    bad = check("4. scoring: the accuracy is the rows', and the paired difference against "
                "fp32 is that accuracy's difference on the same items", ok, bad)

    doc = json.load(open(T.RESULTS))
    p = doc["provenance"]
    ok = (p["kind"] == "harness" and p["binding"] is None and p["proof_sha256"] is None
          and [d["file"] for d in p["datasets"]]
          == [os.path.basename(f) for f in (T.GPQA, T.MMLU, T.IFEVAL)]
          and provenance.rows(doc) == doc["result"])
    bad = check("5. the record carries a harness provenance block: no proof digest, no "
                "binding, and the three datasets named by name and digest", ok, bad)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        T.markdown(res)
    lines = [l for l in out.getvalue().splitlines() if l.startswith("| `")]
    delta, se, nd = E.mcnemar({i: r["pred"] == r["gold"] for i, r in E.rows(T.TASKS, "gpqa", "fp32").items()},
                              {i: r["pred"] == r["gold"] for i, r in E.rows(T.TASKS, "gpqa", "b10-final").items()},
                              docs["gpqa"])
    ok = (len(lines) == 2
          and abs(delta - res["b10-final"]["gpqa"]["paired_vs_fp32"]["delta"]) < 1e-12
          and abs(se - res["b10-final"]["gpqa"]["paired_vs_fp32"]["se"]) < 1e-12
          and 0.0 <= E.se_acc(res["fp32"]["gpqa"]["acc"], n) <= 1.0)
    bad = check("6. the writers: one table row per spec, and the report's paired error "
                "is the scorer's on the same rows", ok, bad)

    T.TASKS = os.path.join(tmp, "tasks-ifeval")
    with mock_tokenizer():
        T.run("ifeval", "fp32", 0, 1, None, 3)
    rows = [json.loads(l) for l in
            open(os.path.join(T.TASKS, "ifeval.fp32.0of1.jsonl"))]
    ok = (len(rows) == n
          and all(r["n_gen"] <= T.MAX_NEW for r in rows)
          and all(r["finished"] == (r["n_gen"] < T.MAX_NEW) for r in rows)
          and all(r["response"] for r in rows if r["n_gen"]))
    bad = check(f"7. the generation path: {n} responses, none past the {T.MAX_NEW}-token "
                f"cap, and each one finished exactly when it stopped before it", ok, bad)
    try:
        import langdetect                                   # noqa: F401
        from lm_eval.tasks.ifeval import instructions_registry  # noqa: F401
        s = T.ifeval_scores(docs["ifeval"], rows)
        ok = all(0.0 <= s[k] <= 1.0 for k in
                 ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose"))
        bad = check("8. the IFEval scorer's four rates", ok, bad)
    except ImportError:
        print("--    8. the IFEval scorer reads lm-eval's instruction checkers and "
              "langdetect, which this environment does not carry: NOT exercised")
    return bad


def cpu_weights(n_new=8):
    """One real forward pass of a tiny model, through the same task code.

    Only when the weights are already in the local cache: this path must never
    be the thing that downloads a model, and CI has none.  It is test 6 above --
    packed decode against one at a time -- and the multiple-choice read, at a
    size that runs on a CPU in seconds.
    """
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(CPU_MODEL, local_files_only=True)
        m = Model(CPU_MODEL, SPECS["fp32"])
    except Exception as e:                 # any of: not cached, no network, no room
        print(f"--    9. {CPU_MODEL} is not in the local cache "
              f"({type(e).__name__}), so no real forward pass ran: NOT exercised")
        return 0
    letters = [tok.encode(L, add_special_tokens=False)[0] for L in T.LETTERS[:4]]
    text = T.mc_text("what colour is the sky?", ["green", "blue", "red", "loud"])
    ids = torch.tensor(T.chat_ids(tok, text) + tok.encode(" (", add_special_tokens=False))
    logits, _ = m.forward(ids)
    lp = torch.log_softmax(logits[-1].float(), -1)[letters]
    prompts = [("a", ids), ("b", ids[:-1])]
    single = {pid: T.generate(m, i, n_new) for pid, i in prompts}
    packed = {}
    T.generate_packed(m, prompts, n_new, 2,
                      lambda pid, out, fin: packed.__setitem__(pid, (out, fin)))
    ok = (bool(torch.isfinite(logits).all()) and 0 <= int(lp.argmax()) < 4
          and all(single[p] == packed[p] for p in single))
    print(f"OK    9. {CPU_MODEL} on the CPU: a multiple-choice read, and packed decode "
          f"identical to one at a time over {n_new} tokens"
          if ok else f"FAIL  9. {CPU_MODEL} on the CPU")
    return 0 if ok else 1


@contextlib.contextmanager
def mock_tokenizer():
    """`tasks.run` loads a tokenizer by name; this is the one it gets here."""
    import transformers
    saved = transformers.AutoTokenizer
    transformers.AutoTokenizer = MockTokenizer
    try:
        yield
    finally:
        transformers.AutoTokenizer = saved


def cpu_selftest(n=6):
    """The eval harness on the CPU: no card, no weights, nothing downloaded.

    Everything it writes goes to a temporary directory, so a run of this never
    touches a session's own `out/tasks`.
    """
    setup()
    tmp = tempfile.mkdtemp(prefix="eval-selftest-")
    t0 = time.time()
    try:
        bad = cpu_tasks(tmp, n)
        bad += cpu_weights()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if not bad else 'FAIL'} ({time.time() - t0:.1f}s); "
          f"the five tests above this path need a card and the 8B weights.")
    return 1 if bad else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cpu", action="store_true",
                   help="the eval harness itself -- the task loaders, the decode the "
                        "task runner drives, the scoring and the writers -- against a "
                        "model with no weights; no card, nothing downloaded")
    p.add_argument("--packed", action="store_true",
                   help="the packed-decode test only, merged into an existing record")
    a = p.parse_args()
    if a.cpu:
        sys.exit(cpu_selftest())
    setup()
    ev, _ = tokens()
    ids = chunks(ev)[0][:N_TOK]
    t0 = time.time()
    if a.packed:
        passed = [test_packed()]
    else:
        passed = [test_vs_hf(ids), test_accumulators(chunks(ev)[0]), test_exp_table(),
                  test_order(chunks(ev)[1][:64]), test_packed()]
    results["passed"] = all(passed)
    results["seconds"] = round(time.time() - t0, 1)
    path = os.path.join(OUT, "selftest.json")
    if a.packed and os.path.exists(path):   # merge into the full run's record
        prev = json.load(open(path))
        prev["packed"] = results["packed"]
        results = prev
    json.dump(results, open(path, "w"), indent=1)
    print(f"{'PASS' if all(passed) else 'FAIL'} ({results['seconds']}s) -> out/selftest.json")
