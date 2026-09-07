"""B8 step 3: the task evals -- GPQA-Diamond, an MMLU-Pro subset, IFEval -- on B1's
forward pass, one spec per invocation.

    . b8/env.sh
    python b8/tasks.py --task gpqa --spec b6-design-p2o
    python b8/tasks.py --task ifeval --spec fp32 --shard 0/4      # generation, sharded
    python b8/tasks.py --score                                     # merge -> out/tasks.json

Multiple choice (GPQA-D, MMLU-Pro) is scored zero-shot from one forward pass: the
question and its lettered choices go in the user turn of the chat template, the
assistant turn is opened with " (" and the answer is the argmax over the letter
tokens at that position -- lm-eval's `gpqa_diamond_zeroshot` text (its seeded
choice shuffle included), restricted-choice, under the instruct model's template.
IFEval is greedy generation to a stop token or 1280 new tokens (lm-eval's
default), scored with the instruction checkers lm-eval ships (the Google
reference implementation), prompt- and instruction-level, strict and loose.
Outputs are one JSONL row per item, appended, so a shard can be resumed.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from harness import provenance
from harness.witness.common import DEVICE, MODEL, OUT, SPECS, THREADS, setup
from harness.witness.model import Cache, Model
from harness.witness.ppl import attention_scales, calibration_scales

DATA = os.environ.get("ZK_EVAL_DATA", os.path.join(HERE, "data"))
TASKS = os.path.join(OUT, "tasks")
RESULTS = os.path.join(OUT, "tasks.json")
GPQA = os.path.join(DATA, "gpqa_jingze/data/test-00000-of-00001.parquet")
MMLU = os.path.join(DATA, "mmlu_pro/data/test-00000-of-00001.parquet")
IFEVAL = os.path.join(DATA, "ifeval/ifeval_input_data.jsonl")
STOP = {128001, 128008, 128009}   # Llama-3.1: <|end_of_text|>, <|eom_id|>, <|eot_id|>
MAX_NEW = 1280                    # lm-eval's ifeval `max_gen_toks`
MMLU_PER_CAT, MMLU_SEED = 100, 0  # 14 categories -> 1,400 questions
LETTERS = "ABCDEFGHIJ"


# -- data -----------------------------------------------------------------
def preprocess(text):
    """lm-eval's gpqa choice preprocessing."""
    if text is None:
        return " "
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def mc_text(question, choices, lead=""):
    opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{lead}What is the correct answer to this question:{question}\nChoices:\n{opts}\nAnswer:"


def load_gpqa():
    import pandas as pd
    df = pd.read_parquet(GPQA)
    rng = random.Random(42)
    docs = []
    for _, r in df.iterrows():
        choices = [preprocess(r["Incorrect Answer 1"]), preprocess(r["Incorrect Answer 2"]),
                   preprocess(r["Incorrect Answer 3"]), preprocess(r["Correct Answer"])]
        rng.shuffle(choices)
        docs.append(dict(id=str(r["Record ID"]), text=mc_text(r["Question"], choices),
                         n=4, gold=choices.index(preprocess(r["Correct Answer"])),
                         group=r["High-level domain"]))
    return docs


def load_mmlu_pro():
    import pandas as pd
    df = pd.read_parquet(MMLU)
    rng = random.Random(MMLU_SEED)
    docs = []
    for cat, g in sorted(df.groupby("category")):
        for j in sorted(rng.sample(range(len(g)), min(MMLU_PER_CAT, len(g)))):
            r = g.iloc[j]
            opts = [o for o in list(r["options"]) if o != "N/A"]
            gold = int(r["answer_index"])
            assert gold < len(opts) and LETTERS[gold] == r["answer"], r["question_id"]
            lead = f"The following is a multiple choice question about {cat}.\n\n"
            docs.append(dict(id=str(r["question_id"]), text=mc_text(r["question"], opts, lead),
                             n=len(opts), gold=gold, group=cat))
    return docs


def load_ifeval():
    docs = []
    for line in open(IFEVAL, encoding="utf-8"):
        r = json.loads(line)
        docs.append(dict(id=str(r["key"]), prompt=r["prompt"],
                         instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"]))
    return docs


LOADERS = {"gpqa": load_gpqa, "mmlu_pro": load_mmlu_pro, "ifeval": load_ifeval}


# -- the model ------------------------------------------------------------
def load_model(spec_name):
    spec = SPECS[spec_name]
    scales = calibration_scales() if spec.linear == "static" else None
    ascales = attention_scales(spec) if spec.int_softmax else None
    return Model(MODEL, spec, static_scales=scales, attn_scales=ascales)


def chat_ids(tok, user):
    text = tok.apply_chat_template([{"role": "user", "content": user}],
                                   add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False).input_ids


def generate(m, ids, max_new):
    """Greedy decode of one prompt: (tokens, finished-on-a-stop-token)."""
    logits, cache = m.forward(ids)
    out = []
    for _ in range(max_new):
        nxt = int(logits[-1].argmax().item())
        if nxt in STOP:
            return out, True
        out.append(nxt)
        logits, cache = m.forward(torch.tensor([nxt]), cache)
    return out, False


def generate_packed(m, prompts, max_new, batch, on_done=None):
    """Greedy decode of many prompts, `batch` at a time in one KV cache.

    Each prompt is prefilled on its own (the ordinary single-sequence forward)
    and merged into the shared cache under its own sequence id; every decode
    step then feeds one token per live sequence, with the `(seq, pos)` mask
    keeping the sequences apart (`Cache`).  A finished sequence's keys are
    dropped and the next prompt admitted, so the batch stays full.  The token
    stream is bit-identical to `generate`'s (b8/selftest.py checks it); only
    the wall time changes, by about `batch`x.
    `prompts`: [(id, ids)]; yields nothing, calls `on_done(id, tokens, finished)`.
    """
    queue = list(prompts)
    cache, active, next_seq = Cache(m.L), [], 0
    while queue or active:
        while len(active) < batch and queue:
            pid, ids = queue.pop(0)
            logits, own = m.forward(ids, Cache(m.L))
            cache.merge(own, next_seq)
            active.append(dict(id=pid, seq=next_seq, pos=len(ids), out=[],
                               nxt=int(logits[-1].argmax().item())))
            next_seq += 1
        live = [a for a in active if a["nxt"] not in STOP and len(a["out"]) < max_new]
        for a in active:
            if a not in live:
                on_done(a["id"], a["out"], a["nxt"] in STOP)
        if len(live) < len(active):
            active = live
            if active:
                cache.keep({a["seq"] for a in active})
            else:
                cache = Cache(m.L)
            continue
        for a in active:
            a["out"].append(a["nxt"])
        dev = m.emb.device
        tok = torch.tensor([a["nxt"] for a in active], device=dev)
        pos = torch.tensor([a["pos"] for a in active], device=dev)
        seq = torch.tensor([a["seq"] for a in active], device=dev)
        logits, _ = m.forward(tok, cache, pos=pos, seq=seq)
        nxt = logits.argmax(-1).tolist()
        for a, t in zip(active, nxt):
            a["nxt"], a["pos"] = int(t), a["pos"] + 1


def run(task, spec_name, shard, n_shards, limit, batch):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = LOADERS[task]()[:limit][shard::n_shards]
    os.makedirs(TASKS, exist_ok=True)
    path = os.path.join(TASKS, f"{task}.{spec_name}.{shard}of{n_shards}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path) if l.strip()}
    todo = [d for d in docs if d["id"] not in done]
    print(f"== {task} / {spec_name} shard {shard}/{n_shards}: {len(docs)} items, "
          f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    t0 = time.time()
    m = load_model(spec_name)
    print(f"   model loaded ({time.time() - t0:.0f}s)", flush=True)
    letter_ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
    assert all(len(x) == 1 for x in letter_ids), letter_ids
    letter_ids = [x[0] for x in letter_ids]
    paren = tok.encode(" (", add_special_tokens=False)
    n_tok, n_done, t1 = 0, 0, time.time()
    with open(path, "a") as f:
        if task == "ifeval":
            prompts = [(d["id"], torch.tensor(chat_ids(tok, d["prompt"]))) for d in todo]
            n_prompt = {d["id"]: len(ids) for d, (_, ids) in zip(todo, prompts)}

            def on_done(pid, out, finished):
                nonlocal n_tok, n_done
                n_tok += len(out)
                n_done += 1
                f.write(json.dumps(dict(id=pid, response=tok.decode(out, skip_special_tokens=True),
                                        n_prompt=n_prompt[pid], n_gen=len(out),
                                        finished=finished)) + "\n")
                f.flush()
                if n_done % 10 == 0 or n_done == len(todo):
                    el = time.time() - t1
                    print(f"   {n_done}/{len(todo)}  {el:6.0f}s  {n_tok / el:5.1f} tok/s", flush=True)
            generate_packed(m, prompts, MAX_NEW, batch, on_done)
            return
        for i, d in enumerate(todo):
            ids = torch.tensor(chat_ids(tok, d["text"]) + paren)
            logits, _ = m.forward(ids)
            lp = torch.log_softmax(logits[-1].float(), -1)[letter_ids[:d["n"]]]
            row = dict(id=d["id"], pred=int(lp.argmax()), gold=d["gold"], group=d["group"],
                       lp=[round(x, 4) for x in lp.tolist()], n_prompt=len(ids))
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"   {i + 1}/{len(todo)}  {time.time() - t1:6.0f}s", flush=True)


# -- scoring --------------------------------------------------------------
def ifeval_scores(docs, rows):
    """lm-eval's `process_results` for ifeval: prompt/instruction level, strict/loose."""
    import langdetect
    from lm_eval.tasks.ifeval import instructions_registry
    # Two sources of scorer noise in the reference checkers, neither pinned by
    # Google's code or lm-eval: `build_description` draws a random value for a
    # missing kwarg (2 of IFEval's 834 instructions, `keywords:letter_frequency`
    # with a null `letter`), and `langdetect` is randomised unless seeded.  Seeded
    # here so every spec faces the same draws and a scoring reproduces.
    random.seed(0)
    langdetect.DetectorFactory.seed = 0
    by_id = {d["id"]: d for d in docs}
    ps, iss, pl, il = [], [], [], []
    for r in rows:
        d = by_id[r["id"]]
        response = r["response"]
        rs = response.split("\n")
        variants = [response, response.replace("*", ""),
                    "\n".join(rs[1:]).strip(), "\n".join(rs[:-1]).strip(), "\n".join(rs[1:-1]).strip()]
        variants += [v.replace("*", "") for v in variants[2:]]
        strict, loose = [], []
        for k, iid in enumerate(d["instruction_id_list"]):
            cls = instructions_registry.INSTRUCTION_DICT[iid]
            inst = cls(iid)
            kwargs = {a: b for a, b in (d["kwargs"][k] or {}).items() if b is not None}
            inst.build_description(**kwargs)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=d["prompt"])
            strict.append(bool(response.strip()) and inst.check_following(response))
            loose.append(any(v.strip() and inst.check_following(v) for v in variants))
        ps.append(all(strict)); pl.append(all(loose))
        iss += strict; il += loose
    mean = lambda xs: sum(xs) / len(xs)
    return dict(prompt_strict=mean(ps), inst_strict=mean(iss), prompt_loose=mean(pl),
                inst_loose=mean(il), n_prompts=len(ps), n_inst=len(iss),
                mean_gen=mean([r["n_gen"] for r in rows]),
                unfinished=sum(1 for r in rows if not r["finished"]))


def score():
    res = {}
    docs = {t: LOADERS[t]() for t in LOADERS}
    pairs = sorted({tuple(os.path.basename(p).split(".")[:2])
                    for p in glob.glob(os.path.join(TASKS, "*.jsonl"))})
    for task, spec_name in pairs:
        rows = {}
        for p in glob.glob(os.path.join(TASKS, f"{task}.{spec_name}.*.jsonl")):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows[r["id"]] = r
        rows = [rows[d["id"]] for d in docs[task] if d["id"] in rows]
        if not rows:
            continue
        complete = len(rows) == len(docs[task])
        if task == "ifeval":
            r = ifeval_scores(docs[task], rows)
        else:
            acc = sum(x["pred"] == x["gold"] for x in rows) / len(rows)
            groups = {}
            for x in rows:
                g = groups.setdefault(x["group"], [0, 0])
                g[0] += x["pred"] == x["gold"]
                g[1] += 1
            r = dict(acc=acc, n=len(rows), by_group={g: v[0] / v[1] for g, v in groups.items()},
                     preds={x["id"]: x["pred"] for x in rows})
        r["complete"] = complete
        res.setdefault(spec_name, {})[task] = r
    # agreement with the float baseline's predictions, per spec
    for spec_name, tasks in res.items():
        for task, r in tasks.items():
            if "preds" in r and "fp32" in res and task in res["fp32"]:
                base = res["fp32"][task]["preds"]
                common = [i for i in r["preds"] if i in base]
                r["agree_fp32"] = sum(r["preds"][i] == base[i] for i in common) / max(1, len(common))
    for tasks in res.values():
        for r in tasks.values():
            r.pop("preds", None)
    json.dump({"provenance": provenance.block(
                   model=MODEL, device=DEVICE, threads=THREADS,
                   module="harness.eval.tasks",
                   specs_in_file=sorted(res),
                   datasets=[provenance.digest(f) for f in (GPQA, MMLU, IFEVAL)]),
               "result": res}, open(RESULTS, "w"), indent=1, sort_keys=True)
    return res


def markdown(res):
    cols = ["gpqa", "mmlu_pro", "ifeval"]
    print("| spec | GPQA-D acc (n) | agree fp32 | MMLU-Pro acc (n) | agree fp32 | "
          "IFEval prompt-strict | inst-strict | prompt-loose | inst-loose | gen tok |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for spec_name in [s for s in SPECS if s in res]:
        r = res[spec_name]
        cells = []
        for t in cols[:2]:
            if t in r:
                x = r[t]
                cells += [f"{x['acc'] * 100:.1f}% ({x['n']}){'' if x['complete'] else ' *partial*'}",
                          f"{x['agree_fp32'] * 100:.1f}%" if "agree_fp32" in x else "--"]
            else:
                cells += ["--", "--"]
        if "ifeval" in r:
            x = r["ifeval"]
            cells += [f"{x[k] * 100:.1f}%" for k in ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose")]
            cells += [f"{x['mean_gen']:.0f}{'' if x['complete'] else ' *partial*'}"]
        else:
            cells += ["--"] * 5
        print(f"| `{spec_name}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(LOADERS))
    p.add_argument("--spec", choices=list(SPECS))
    p.add_argument("--shard", default="0/1", help="i/N: this process takes items i::N")
    p.add_argument("--limit", type=int, default=None, help="smoke run: first N items only")
    p.add_argument("--batch", type=int, default=32, help="ifeval: sequences decoded together")
    p.add_argument("--score", action="store_true", help="merge shards and print the table")
    a = p.parse_args()
    setup()
    if a.score:
        markdown(score())
        raise SystemExit
    i, n = (int(x) for x in a.shard.split("/"))
    run(a.task, a.spec, i, n, a.limit, a.batch)
