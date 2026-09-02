"""Stream E5v: the task suite under `y-delay-vw12`, read against its controls.

    cd experiments && E5V_OUT=... E5V_REF=... python3 b8/e5v/report.py

`b8/e5t/report.py` with the spec under test changed and one comparison added.
It answers the same three questions E5t's did -- does this session's `fp32`
control reproduce E5t's item for item, what is the *paired* (McNemar) error on
each spec-vs-control difference, and where does a one-point aggregate hide by
domain -- plus the exact IFEval generated-token sum, which `tasks.json` carries
only as a rounded mean.

`E5V_OUT` is this session's `B1_OUT` (`b8/out/e5v`), `E5V_REF` E5t's committed
rows (`b8/out/e5t`), which were scored on this box on these items.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(HERE, os.pardir, "e5t"))
import tasks as T  # noqa: E402  (b8/tasks.py: the loaders and the ifeval scorer)
from report import ifeval_items, mcnemar, rows, se_acc  # noqa: E402  (b8/e5t/report.py)

E5V = os.environ.get("E5V_OUT", os.path.join(HERE, os.pardir, "out", "e5v"))
E5T = os.environ.get("E5V_REF", os.path.join(HERE, os.pardir, "out", "e5t"))
SPEC = "y-delay-vw12"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def delta_line(label, a, b, docs):
    """One row of a paired table: b - a, with the unpaired and the paired SE.

    Counts beside every percentage, and the discordant pairs split into gained
    and lost, because the percentages hide both."""
    d, se, nd = mcnemar(a, b, docs)
    ids = [x["id"] for x in docs if x["id"] in a and x["id"] in b]
    gained = sum(1 for i in ids if not a[i] and b[i])
    lost = sum(1 for i in ids if a[i] and not b[i])
    ca, cb, n = sum(a.values()), sum(b.values()), len(a)
    ma, mb = ca / n, cb / n
    return (f"| {label} | {ma * 100:.1f}% ({ca}) | {mb * 100:.1f}% ({cb}) | "
            f"{d * 100:+.1f} pts | ±{se_acc(ma, n) * 100:.1f}% | ±{se * 100:.2f}% "
            f"({abs(d) / se if se else 0:.1f}σ) | {nd} ({gained} gained / {lost} lost) |")


def ok_map(d, task, spec):
    r = rows(os.path.join(d, "tasks"), task, spec)
    return {i: x["pred"] == x["gold"] for i, x in r.items()}


def main():
    docs = {t: T.LOADERS[t]() for t in T.LOADERS}
    e5v = json.load(open(os.path.join(E5V, "tasks.json")))
    e5t = json.load(open(os.path.join(E5T, "tasks.json")))

    print("== 1. the environment control: this session's `fp32` against E5t's\n")
    print(f"{'task':<22} {'E5t (H200)':>11} {'E5v (H200)':>11}  items identical   file sha256")
    for task in ("gpqa", "mmlu_pro"):
        old, new = e5t["fp32"][task], e5v["fp32"][task]
        a = rows(os.path.join(E5T, "tasks"), task, "fp32")
        c = rows(os.path.join(E5V, "tasks"), task, "fp32")
        same = sum(1 for i in c if i in a and a[i]["pred"] == c[i]["pred"])
        samelp = sum(1 for i in c if i in a and a[i]["lp"] == c[i]["lp"])
        f = f"{task}.fp32.0of1.jsonl"
        bytes_same = sha(os.path.join(E5T, "tasks", f)) == sha(os.path.join(E5V, "tasks", f))
        print(f"{task:<22} {old['acc'] * 100:10.1f}% {new['acc'] * 100:10.1f}%  "
              f"{same}/{len(c)} pred, {samelp}/{len(c)} logprob   "
              f"{'identical' if bytes_same else 'DIFFERENT'}")
    old, new = e5t["fp32"]["ifeval"], e5v["fp32"]["ifeval"]
    a = rows(os.path.join(E5T, "tasks"), "ifeval", "fp32")
    c = rows(os.path.join(E5V, "tasks"), "ifeval", "fp32")
    same = sum(1 for i in c if i in a and a[i]["response"] == c[i]["response"])
    shards = [f"ifeval.fp32.{i}of2.jsonl" for i in range(2)]
    bytes_same = all(os.path.exists(os.path.join(E5T, "tasks", f))
                     and os.path.exists(os.path.join(E5V, "tasks", f))
                     and sha(os.path.join(E5T, "tasks", f)) == sha(os.path.join(E5V, "tasks", f))
                     for f in shards)
    print(f"{'ifeval (p-strict)':<22} {old['prompt_strict'] * 100:10.1f}% "
          f"{new['prompt_strict'] * 100:10.1f}%  {same}/{len(c)} responses        "
          f"{'identical' if bytes_same else 'DIFFERENT'}")

    print(f"\n== 2. `{SPEC}` against its same-session `fp32` control\n")
    print("| task | fp32 | " + SPEC + " | delta | unpaired SE | paired SE (McNemar) | discordant |")
    print("|---|---|---|---|---|---|---|")
    for task, label in (("gpqa", "GPQA-D (198)"), ("mmlu_pro", "MMLU-Pro (1,400)")):
        print(delta_line(label, ok_map(E5V, task, "fp32"), ok_map(E5V, task, SPEC),
                         docs[task]))
    items = {}
    for where, spec in ((E5V, "fp32"), (E5V, SPEC), (E5T, "b10-final")):
        r = rows(os.path.join(where, "tasks"), "ifeval", spec)
        ordered = [r[d["id"]] for d in docs["ifeval"]]
        items[spec] = ifeval_items(docs["ifeval"], ordered)
        got = T.ifeval_scores(docs["ifeval"], ordered)
        for k, v in items[spec].items():
            assert abs(sum(v.values()) / len(v) - got[k]) < 1e-12, (spec, k)
    for k, label in (("prompt_strict", "IFEval prompt-strict (541)"),
                     ("inst_strict", "IFEval inst-strict (834)"),
                     ("prompt_loose", "IFEval prompt-loose (541)"),
                     ("inst_loose", "IFEval inst-loose (834)")):
        a, b = items["fp32"][k], items[SPEC][k]
        print(delta_line(label, a, b, [dict(id=i) for i in a]))

    fr = rows(os.path.join(E5V, "tasks"), "ifeval", "fp32")
    sr = rows(os.path.join(E5V, "tasks"), "ifeval", SPEC)
    br = rows(os.path.join(E5T, "tasks"), "ifeval", "b10-final")
    same = sum(1 for i in sr if i in fr and fr[i]["response"] == sr[i]["response"])
    print(f"\nIFEval generations identical to fp32's: {same}/{len(sr)}; "
          f"mean gen tok {e5v['fp32']['ifeval']['mean_gen']:.1f} -> "
          f"{e5v[SPEC]['ifeval']['mean_gen']:.1f}; "
          f"unfinished at the 1,280-token cap {e5v['fp32']['ifeval']['unfinished']} -> "
          f"{e5v[SPEC]['ifeval']['unfinished']}")
    print("exact IFEval generated-token sums (the analogue of "
          "`suite.ifeval.generated_tokens`):")
    for name, r in ((SPEC, sr), ("fp32", fr), ("b10-final (E5t)", br)):
        print(f"   {name:<22} {sum(x['n_gen'] for x in r.values()):>8,} tokens "
              f"over {len(r)} prompts")

    print(f"\n== 3. `{SPEC}` against `b10-final` (E5t's rows, same box, same items)\n")
    print("| task | b10-final | " + SPEC + " | delta | unpaired SE | "
          "paired SE (McNemar) | discordant |")
    print("|---|---|---|---|---|---|---|")
    for task, label in (("gpqa", "GPQA-D (198)"), ("mmlu_pro", "MMLU-Pro (1,400)")):
        print(delta_line(label, ok_map(E5T, task, "b10-final"), ok_map(E5V, task, SPEC),
                         docs[task]))
    for k, label in (("prompt_strict", "IFEval prompt-strict (541)"),
                     ("inst_strict", "IFEval inst-strict (834)"),
                     ("prompt_loose", "IFEval prompt-loose (541)"),
                     ("inst_loose", "IFEval inst-loose (834)")):
        print(delta_line(label, items["b10-final"][k], items[SPEC][k],
                         [dict(id=i) for i in items["b10-final"][k]]))

    print("\n== 4. the three specs side by side (E5t's rows for `b10-final`)\n")
    print("| spec | GPQA-D | MMLU-Pro | IFEval p-strict | i-strict | p-loose | i-loose |"
          " mean gen | ppl vs fp32 |")
    print("|---|---|---|---|---|---|---|---|---|")
    # both from stream Z's recorded rows, which this session's gate reproduces
    p32 = json.load(open(os.path.join(E5V, "ref", "ppl.json")))
    ppl = {"fp32": None}
    for s in ("b10-final", SPEC):
        ppl[s] = (p32[s]["ppl"] / p32["fp32"]["ppl"] - 1) * 100
    for spec, src in (("fp32", e5v), ("b10-final", e5t), (SPEC, e5v)):
        r = src[spec]
        p = ppl[spec]
        print(f"| `{spec}` | {r['gpqa']['acc'] * 100:.1f}% | {r['mmlu_pro']['acc'] * 100:.1f}% | "
              + " | ".join(f"{r['ifeval'][k] * 100:.1f}%" for k in
                           ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose"))
              + f" | {r['ifeval']['mean_gen']:.1f} | "
              + ("--" if p is None else f"+{p:.3f}%") + " |")
    print("\nitem counts: " + ", ".join(
        f"{spec}/{task} {src[spec][task].get('n', src[spec][task].get('n_prompts'))}"
        f"{'' if src[spec][task]['complete'] else ' PARTIAL'}"
        for spec, src in (("fp32", e5v), ("b10-final", e5t), (SPEC, e5v))
        for task in ("gpqa", "mmlu_pro", "ifeval")))

    print(f"\n== 5. per domain, `{SPEC}` against fp32 -- counts, not percentages\n")
    for task in ("gpqa", "mmlu_pro"):
        cnt = {}
        for spec in ("fp32", SPEC):
            for x in rows(os.path.join(E5V, "tasks"), task, spec).values():
                c = cnt.setdefault(x["group"], dict(n=0, fp32=0))
                if spec == "fp32":
                    c["n"] += 1
                    c["fp32"] += x["pred"] == x["gold"]
                else:
                    c[SPEC] = c.get(SPEC, 0) + (x["pred"] == x["gold"])
        worst = sorted(cnt, key=lambda g: -abs(cnt[g][SPEC] - cnt[g]["fp32"]))
        print(task + ": " + ", ".join(
            f"{g} {cnt[g]['fp32']}->{cnt[g][SPEC]} of {cnt[g]['n']}" for g in worst))


if __name__ == "__main__":
    main()
