"""Stream E5t: the frozen spec's task rows, read against their controls.

    cd experiments && python3 b8/e5t/report.py

Three things the merged `tasks.json` cannot say on its own:

* whether this session's `fp32` and `b10-design` rows *reproduce* B8's and
  B10's recorded ones -- the environment control, item by item, not just at
  the accuracy;
* the **paired** standard error of a spec-vs-fp32 difference.  A single
  accuracy on GPQA-D's 198 items carries +/-3.3%, but the two specs answer the
  *same* questions, so the difference has its own, much smaller, error: only
  the discordant items carry information (McNemar).  B8 quoted the unpaired
  figure beside every task delta; this adds the paired one, the way `ppl.py`
  already does for perplexity;
* the per-domain view, which is where a 1-point aggregate can hide.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from harness.eval import tasks as T

# `ZK_TASKS_OUT` holds this session's `tasks/` + `tasks.json`, `ZK_TASKS_REF` the
# recorded ones (B8's H100 rows).  Defaults are the committed repo layout; the
# box overrides them, since the box tree carries only this session's artifacts.
E5T = os.environ.get("ZK_TASKS_OUT", os.path.join(HERE, os.pardir, "out", "e5t"))
B8 = os.environ.get("ZK_TASKS_REF", os.path.join(HERE, os.pardir, "out"))


def rows(d, task, spec):
    out = {}
    for f in sorted(os.listdir(d)):
        p = f.split(".")
        if len(p) == 4 and p[0] == task and p[1] == spec and p[3] == "jsonl":
            for line in open(os.path.join(d, f)):
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r
    return out


def ifeval_items(docs, rows):
    """`tasks.ifeval_scores`, but keeping the per-prompt and per-instruction
    booleans so an IFEval difference gets a paired error too.  Same seeding and
    the same row order as the shipped scorer, and `main` asserts the four means
    against it, so this is that scorer with its intermediate values kept."""
    import langdetect
    from lm_eval.tasks.ifeval import instructions_registry
    random.seed(0)
    langdetect.DetectorFactory.seed = 0
    by_id = {d["id"]: d for d in docs}
    ps, iss, pl, il = {}, {}, {}, {}
    for r in rows:
        d = by_id[r["id"]]
        response = r["response"]
        rs = response.split("\n")
        variants = [response, response.replace("*", ""),
                    "\n".join(rs[1:]).strip(), "\n".join(rs[:-1]).strip(),
                    "\n".join(rs[1:-1]).strip()]
        variants += [v.replace("*", "") for v in variants[2:]]
        strict, loose = [], []
        for k, iid in enumerate(d["instruction_id_list"]):
            inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
            kwargs = {a: b for a, b in (d["kwargs"][k] or {}).items() if b is not None}
            inst.build_description(**kwargs)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=d["prompt"])
            strict.append(bool(response.strip()) and inst.check_following(response))
            loose.append(any(v.strip() and inst.check_following(v) for v in variants))
        ps[d["id"]] = all(strict)
        pl[d["id"]] = all(loose)
        for k in range(len(strict)):
            iss[f"{d['id']}#{k}"] = strict[k]
            il[f"{d['id']}#{k}"] = loose[k]
    return dict(prompt_strict=ps, prompt_loose=pl, inst_strict=iss, inst_loose=il)


def se_acc(p, n):
    return math.sqrt(p * (1 - p) / n)


def mcnemar(a, b, docs):
    """Paired difference b-a on a shared item set: (delta, paired SE, n_discordant)."""
    xs = [(int(a[d["id"]]), int(b[d["id"]])) for d in docs if d["id"] in a and d["id"] in b]
    n = len(xs)
    disc01 = sum(1 for x, y in xs if not x and y)
    disc10 = sum(1 for x, y in xs if x and not y)
    delta = (disc01 - disc10) / n
    var = (disc01 + disc10 - (disc01 - disc10) ** 2 / n) / n ** 2
    return delta, math.sqrt(max(var, 0.0)), disc01 + disc10


def main():
    docs = {t: T.LOADERS[t]() for t in T.LOADERS}
    e5t = json.load(open(os.path.join(E5T, "tasks.json")))
    b8 = json.load(open(os.path.join(B8, "tasks.json")))

    print("== 1. the environment control: this session's rows against the recorded ones\n")
    print(f"{'spec / task':<26} {'recorded':>10} {'E5t (H200)':>12}  items identical")
    for spec in ("fp32", "b10-design"):
        for task in ("gpqa", "mmlu_pro"):
            old, new = b8[spec][task], e5t[spec][task]
            a = rows(os.path.join(B8, "tasks"), task, spec)
            c = rows(os.path.join(E5T, "tasks"), task, spec)
            same = sum(1 for i in c if i in a and a[i]["pred"] == c[i]["pred"])
            note = f"{same}/{len(c)}" if a else "(no recorded jsonl)"
            print(f"{spec + ' / ' + task:<26} {old['acc'] * 100:9.1f}% {new['acc'] * 100:11.1f}%  {note}")
        old, new = b8[spec]["ifeval"], e5t[spec]["ifeval"]
        a = rows(os.path.join(B8, "tasks"), "ifeval", spec)
        c = rows(os.path.join(E5T, "tasks"), "ifeval", spec)
        same = sum(1 for i in c if i in a and a[i]["response"] == c[i]["response"])
        note = f"{same}/{len(c)} responses" if a else "(no recorded jsonl)"
        print(f"{spec + ' / ifeval-ps':<26} {old['prompt_strict'] * 100:9.1f}% "
              f"{new['prompt_strict'] * 100:11.1f}%  {note}")

    print("\n== 2. `b10-final` against its same-session `fp32` control\n")
    print("| task | fp32 | b10-final | delta | unpaired SE | paired SE (McNemar) | discordant | agree |")
    print("|---|---|---|---|---|---|---|---|")
    for task, label in (("gpqa", "GPQA-D (198)"), ("mmlu_pro", "MMLU-Pro (1,400)")):
        f = rows(os.path.join(E5T, "tasks"), task, "fp32")
        s = rows(os.path.join(E5T, "tasks"), task, "b10-final")
        ok_f = {i: r["pred"] == r["gold"] for i, r in f.items()}
        ok_s = {i: r["pred"] == r["gold"] for i, r in s.items()}
        d, se, nd = mcnemar(ok_f, ok_s, docs[task])
        af, as_ = e5t["fp32"][task]["acc"], e5t["b10-final"][task]["acc"]
        n = e5t["fp32"][task]["n"]
        print(f"| {label} | {af * 100:.1f}% | {as_ * 100:.1f}% | {d * 100:+.1f} pts | "
              f"±{se_acc(af, n) * 100:.1f}% | ±{se * 100:.2f}% ({abs(d) / se if se else 0:.1f}σ) | "
              f"{nd} | {e5t['b10-final'][task]['agree_fp32'] * 100:.1f}% |")
    items = {}
    for spec in ("fp32", "b10-final", "b10-design"):
        r = rows(os.path.join(E5T, "tasks"), "ifeval", spec)
        items[spec] = ifeval_items(docs["ifeval"], [r[d["id"]] for d in docs["ifeval"]])
        got = T.ifeval_scores(docs["ifeval"], [r[d["id"]] for d in docs["ifeval"]])
        for k, v in items[spec].items():
            assert abs(sum(v.values()) / len(v) - got[k]) < 1e-12, (spec, k)
    items["b6-design-p2o"] = ifeval_items(
        docs["ifeval"], [rows(os.path.join(B8, "tasks"), "ifeval", "b6-design-p2o")[d["id"]]
                         for d in docs["ifeval"]])
    keys = [d["id"] for d in docs["ifeval"]]
    for k, label, n in (("prompt_strict", "IFEval prompt-strict (541)", 541),
                        ("inst_strict", "IFEval inst-strict (834)", 834),
                        ("prompt_loose", "IFEval prompt-loose (541)", 541),
                        ("inst_loose", "IFEval inst-loose (834)", 834)):
        a, b = items["fp32"][k], items["b10-final"][k]
        ks = [dict(id=i) for i in a]
        d, se, nd = mcnemar(a, b, ks)
        ma, mb = sum(a.values()) / len(a), sum(b.values()) / len(b)
        print(f"| {label} | {ma * 100:.1f}% | {mb * 100:.1f}% | {d * 100:+.1f} pts | "
              f"±{se_acc(ma, n) * 100:.1f}% | ±{se * 100:.2f}% ({abs(d) / se if se else 0:.1f}σ) | "
              f"{nd} | -- |")
    fr = rows(os.path.join(E5T, "tasks"), "ifeval", "fp32")
    sr = rows(os.path.join(E5T, "tasks"), "ifeval", "b10-final")
    same = sum(1 for i in sr if i in fr and fr[i]["response"] == sr[i]["response"])
    print(f"\nIFEval generations identical to fp32's: {same}/{len(sr)}; "
          f"mean gen tok {e5t['fp32']['ifeval']['mean_gen']:.0f} -> "
          f"{e5t['b10-final']['ifeval']['mean_gen']:.0f}; "
          f"unfinished at the 1,280-token cap {e5t['fp32']['ifeval']['unfinished']} -> "
          f"{e5t['b10-final']['ifeval']['unfinished']}")

    print("\n== 3. `b10-final` against the specs the paper's rows were scored on\n")
    print("| spec | GPQA-D | MMLU-Pro | IFEval p-strict | i-strict | p-loose | i-loose | ppl vs fp32 |")
    print("|---|---|---|---|---|---|---|---|")
    ppl = {"fp32": None, "b6-design-p2o": 8.81, "b10-design": 8.131, "b10-final": 7.846}
    for spec, src in (("fp32", e5t), ("b6-design-p2o", b8), ("b10-design", e5t), ("b10-final", e5t)):
        r = src[spec]
        p = ppl[spec]
        print(f"| `{spec}` | {r['gpqa']['acc'] * 100:.1f}% | {r['mmlu_pro']['acc'] * 100:.1f}% | "
              + " | ".join(f"{r['ifeval'][k] * 100:.1f}%" for k in
                           ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose"))
              + f" | {'--' if p is None else f'+{p:.2f}%'} |")

    print("\n   paired, item by item, against the two specs the paper's rows carry")
    print("   (`fp32` reproduces B8's rows exactly, so the H100 artifacts pair item-wise):\n")
    print("| pair | GPQA-D | MMLU-Pro | IFEval p-strict | IFEval i-strict |")
    print("|---|---|---|---|---|")
    for other, where in (("b6-design-p2o", B8), ("b10-design", E5T)):
        cells = []
        for task in ("gpqa", "mmlu_pro"):
            o = rows(os.path.join(where, "tasks"), task, other)
            n_ = rows(os.path.join(E5T, "tasks"), task, "b10-final")
            d, se, nd = mcnemar({i: r["pred"] == r["gold"] for i, r in o.items()},
                                {i: r["pred"] == r["gold"] for i, r in n_.items()}, docs[task])
            cells.append(f"{d * 100:+.1f} ± {se * 100:.2f} ({abs(d) / se if se else 0:.1f}σ)")
        for k in ("prompt_strict", "inst_strict"):
            a, b = items[other][k], items["b10-final"][k]
            d, se, nd = mcnemar(a, b, [dict(id=i) for i in a])
            cells.append(f"{d * 100:+.1f} ± {se * 100:.2f} ({abs(d) / se if se else 0:.1f}σ)")
        print(f"| `b10-final` − `{other}` | " + " | ".join(cells) + " |")

    print("\n== 4. per domain, `b10-final` against fp32\n")
    for task in ("gpqa", "mmlu_pro"):
        f, s = e5t["fp32"][task]["by_group"], e5t["b10-final"][task]["by_group"]
        worst = sorted(f, key=lambda g: -abs(s[g] - f[g]))
        print(task + ": " + ", ".join(
            f"{g} {f[g] * 100:.0f}->{s[g] * 100:.0f}" for g in worst))


if __name__ == "__main__":
    main()
