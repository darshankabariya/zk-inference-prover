"""E2h: the decode backends.

Two of them, on purpose.  The 500,000x claim's denominator (668 tok/s for
Llama-13B) is an OPTIMIZED, BATCHED figure; measuring ourselves only against a
naive `generate()` loop would flatter us in exactly the way the paper accuses
that estimate of doing.  So:

  hf    -- `transformers.generate()`: what a non-specialist gets.
  vllm  -- a production serving engine: what the belief's denominator means.

Both are measured by the SAME protocol (see `measure`): a prefill-only call,
then a full call for N tokens, the decode's own seconds by difference.
Identical protocol on both backends is itself a like-for-like property -- the
two rows differ in engine, in nothing else.

A backend that is not installed reports `available: false` with the import error.
It never returns a number.

A third backend, `mock`, exists only to exercise the harness where no card and
no model are available.  It produces SYNTHETIC seconds and every run it writes
is stamped `synthetic: true`, which `lfl.py` treats as a BLOCKING dimension.
A synthetic number can never reach a table.
"""

import hashlib
import os
import time

TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "witness", "text.txt")
# Distinct prompts per batch element: no engine-level prefix cache can turn a
# batch of B sequences into one prefill and inflate the batched rate.
BATCH_STRIDE = 1013


def _sha(ids):
    h = hashlib.sha256()
    for row in ids:
        h.update((",".join(str(int(x)) for x in row)).encode())
    return h.hexdigest()[:16]


def build_prompts(tok, context, batch):
    """`batch` prompts of exactly `context` tokens each, deterministic, distinct."""
    with open(TEXT, encoding="utf-8") as f:
        corpus = f.read()
    need = BATCH_STRIDE * batch + context + 8
    ids = tok(corpus, add_special_tokens=False)["input_ids"]
    while len(ids) < need:
        ids = ids + ids
    bos = getattr(tok, "bos_token_id", None)
    out = []
    for i in range(batch):
        o = i * BATCH_STRIDE
        row = ids[o:o + context]
        if bos is not None:
            row = [bos] + row[: context - 1]
        assert len(row) == context, (len(row), context)
        out.append(row)
    # The distinctness claim is CHECKED, not asserted in prose: `text.txt` is
    # tiled to reach the needed length, and a corpus short enough for the tiling
    # period to line up with BATCH_STRIDE would hand two batch elements the same
    # prompt -- exactly the prefix-cache collapse this guards against.  Fail
    # loudly rather than publish an inflated batched rate.
    if len({tuple(r) for r in out}) != batch:
        raise RuntimeError(
            f"build_prompts produced duplicate prompts at batch={batch}, "
            f"context={context}: text.txt tokenizes to {len(ids)} ids and the "
            f"stride is {BATCH_STRIDE}. A duplicated prompt lets an engine-level "
            "prefix cache collapse the batch and inflate the aggregate rate.")
    return out


class Unavailable(RuntimeError):
    pass


# --- HuggingFace ----------------------------------------------------------

class HFBackend:
    name = "hf"
    label = "transformers.generate() -- the non-specialist path"

    def __init__(self, model_id, revision, dtype, device, attn):
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise Unavailable(f"transformers/torch not importable: {e}")
        self.torch, self.tf = torch, transformers
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        td = getattr(torch, dtype)
        kw = dict(revision=revision, dtype=td)
        if attn:
            kw["attn_implementation"] = attn
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model = self.model.to(device).eval()
        self.attn = getattr(self.model.config, "_attn_implementation", attn)
        self.version = f"transformers {transformers.__version__}, torch {torch.__version__}"

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def generate(self, prompts, n_new):
        torch = self.torch
        ids = torch.tensor(prompts, dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        cfg = self.tf.GenerationConfig(
            max_new_tokens=n_new, min_new_tokens=n_new,
            do_sample=False, num_beams=1, use_cache=True,
            # EOS suppressed: the proved statement covers a fixed 512 positions,
            # so the baseline must produce exactly as many, not stop early.
            eos_token_id=None, stop_strings=None,
            pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None
            else (self.tok.eos_token_id if self.tok.eos_token_id is not None else 0),
        )
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
        got = out.shape[1] - ids.shape[1]
        if got != n_new:
            raise RuntimeError(f"generated {got} tokens, asked for {n_new}")
        return [row[ids.shape[1]:].tolist() for row in out]

    def info(self):
        cfg = self.model.config
        return {"n_layers": getattr(cfg, "num_hidden_layers", None),
                "n_heads": getattr(cfg, "num_attention_heads", None),
                "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
                "hidden": getattr(cfg, "hidden_size", None),
                "vocab": getattr(cfg, "vocab_size", None),
                "attn_implementation": self.attn}


# --- vLLM -----------------------------------------------------------------

class VLLMBackend:
    name = "vllm"
    label = "vLLM -- the production serving path"

    def __init__(self, model_id, revision, dtype, device, attn, **kw):
        try:
            import vllm
        except ImportError as e:
            raise Unavailable(
                f"vllm not importable ({e}). vLLM ships no CPU-only wheel that "
                "co-exists with this project's CPU torch pin, so it is NOT a "
                "dependency of experiments/pyproject.toml -- and inside "
                "experiments/ the pin makes even a resolve fail: uv backtracks "
                "past every wheel to the vllm==0.1.3 sdist and tries to build "
                "it. Install it into its OWN venv, off the default index: "
                "`sh e2/run.sh vllmenv`, then run with that interpreter "
                "(VLLM_PYTHON=.venv-vllm/bin/python).")
        self.vllm = vllm
        self.version = f"vllm {vllm.__version__}"
        from vllm import LLM
        self.llm = LLM(
            model=model_id, revision=revision, dtype=dtype,
            # No prefix caching: it would collapse a batch's prefills and inflate
            # the batched decode rate we are about to publish.
            enable_prefix_caching=False,
            max_model_len=kw.get("max_model_len"),
            gpu_memory_utilization=kw.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=kw.get("tensor_parallel_size", 1),
            enforce_eager=kw.get("enforce_eager", False),
        )
        self.tok = self.llm.get_tokenizer()
        self.device = device
        # vLLM does not run a "batch": it runs a SCHEDULER.  A nominal batch of
        # B is OFFERED concurrency; the engine may serve it in waves bounded by
        # max_num_seqs and by how much KV fits.  Aggregate throughput is still
        # measured correctly (wall over all tokens produced), but the recorded
        # `batch_size` is offered concurrency, not a guaranteed in-flight count
        # -- so the limits travel in the JSON and a reader can see it.
        try:
            sc = self.llm.llm_engine.vllm_config.scheduler_config
            cc = self.llm.llm_engine.vllm_config.cache_config
            self.sched = {"max_num_seqs": getattr(sc, "max_num_seqs", None),
                          "max_num_batched_tokens": getattr(sc, "max_num_batched_tokens", None),
                          "max_model_len": getattr(sc, "max_model_len", None),
                          "enable_prefix_caching": getattr(cc, "enable_prefix_caching", None),
                          "block_size": getattr(cc, "block_size", None)}
        except Exception as e:  # a vLLM version that moved the config
            self.sched = {"error": f"{type(e).__name__}: {e}"}

    def sync(self):
        pass

    def generate(self, prompts, n_new):
        from vllm import SamplingParams
        try:
            from vllm import TokensPrompt
        except ImportError:
            from vllm.inputs import TokensPrompt
        # detokenize=False: the hf arm returns token ids and never runs the
        # tokenizer inside the timed region, so vLLM must not either. Otherwise
        # the "difference between the two rows is engine and nothing else" claim
        # is false by one detokenization pass.
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                            max_tokens=n_new, min_tokens=n_new, ignore_eos=True,
                            n=1, seed=0, detokenize=False)
        reqs = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]
        outs = self.llm.generate(reqs, sp)
        got = [len(o.outputs[0].token_ids) for o in outs]
        if any(g != n_new for g in got):
            raise RuntimeError(f"generated {got}, asked for {n_new}")
        return [list(o.outputs[0].token_ids) for o in outs]

    def info(self):
        cfg = self.llm.llm_engine.model_config.hf_config
        return {"n_layers": getattr(cfg, "num_hidden_layers", None),
                "n_heads": getattr(cfg, "num_attention_heads", None),
                "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
                "hidden": getattr(cfg, "hidden_size", None),
                "vocab": getattr(cfg, "vocab_size", None),
                "attn_implementation": "vllm-internal",
                "scheduler": self.sched,
                "batch_semantics": "OFFERED concurrency; vLLM schedules it -- see "
                                   "scheduler.max_num_seqs"}


# --- mock -----------------------------------------------------------------

class MockBackend:
    """No model, no card, no number anyone may quote.

    Exists so the parts of this harness that only a full 8B run would otherwise
    exercise -- the batch sweep and its OOM-is-a-result path, the exact-token
    assertion, the two-call timing protocol and both counting conventions -- can
    be run on a laptop in seconds.  Its seconds come from a crude arithmetic
    model (a per-token cost plus a per-key attention term); they are NOT a
    prediction of anything.  It OOMs on a configurable KV budget so the sweep's
    stop path is real code.

    Everything it writes carries `synthetic: true`, and `lfl.check` makes that a
    BLOCKING dimension.
    """

    name = "mock"
    label = "SYNTHETIC -- harness exercise only, never a measurement"
    synthetic = True
    version = "mock/1"

    class _Tok:
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            # a stable, cheap pseudo-tokenization: one id per 4 characters
            return {"input_ids": [((ord(text[i]) * 31 + i) % 32000) + 3
                                  for i in range(0, len(text), 4)]}

        def __len__(self):
            return 32000

    def __init__(self, model_id, revision, dtype, device, attn, **kw):
        self.tok = self._Tok()
        self.device = device
        self.model_id = model_id
        # a toy cost model; the constants are arbitrary and labelled as such
        self.s_per_token = float(os.environ.get("ZK_BASELINE_MOCK_S_PER_TOKEN", "0.002"))
        self.s_per_key = float(os.environ.get("ZK_BASELINE_MOCK_S_PER_KEY", "1e-7"))
        self.kv_budget_tokens = int(os.environ.get("ZK_BASELINE_MOCK_KV_BUDGET", "200000"))

    def sync(self):
        pass

    def generate(self, prompts, n_new):
        b, c = len(prompts), len(prompts[0])
        if b * (c + n_new) > self.kv_budget_tokens:
            raise RuntimeError(
                f"CUDA out of memory (SYNTHETIC): {b} x {c + n_new} tokens over "
                f"the mock KV budget of {self.kv_budget_tokens}")
        per_step = self.s_per_token + self.s_per_key * (c + n_new / 2)
        conc = 1 + 0.25 * (b - 1)          # batching is sublinear, crudely
        time.sleep(per_step * conc * (1 + (n_new - 1) * 0.5))
        return [[(p[-1] + i) % 32000 for i in range(n_new)] for p in prompts]

    def info(self):
        return {"n_layers": None, "n_heads": None, "n_kv_heads": None,
                "hidden": None, "vocab": 32000, "attn_implementation": "none",
                "SYNTHETIC": True,
                "cost_model": {"s_per_token": self.s_per_token,
                               "s_per_key": self.s_per_key,
                               "kv_budget_tokens": self.kv_budget_tokens},
                "warning": "SYNTHETIC seconds from an arbitrary arithmetic model. "
                           "Not a measurement of any model on any hardware."}


BACKENDS = {"hf": HFBackend, "vllm": VLLMBackend, "mock": MockBackend}


def probe(name):
    """Is this backend importable here?  Reported, never faked."""
    try:
        if name == "hf":
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return {"available": True, "reason": None}
        if name == "vllm":
            import vllm  # noqa: F401
            return {"available": True, "reason": None}
        if name == "mock":
            return {"available": True, "reason": None,
                    "warning": "SYNTHETIC. Exercises the harness; produces no measurement."}
    except ImportError as e:
        return {"available": False, "reason": str(e)}
    return {"available": False, "reason": f"unknown backend {name}"}


# --- the protocol both backends are measured by ---------------------------

def measure(be, prompts, n_new, reps, warmup):
    """The three wall-clock times of one repetition.  Identical on every backend.

    A warmup pass, then per repetition two calls:

      a `max_new_tokens=1` call, whose wall-clock time is the PREFILL's -- it
      reads the whole prompt and emits the first generated position;

      a full `n_new`-token call, whose wall-clock time is the pass END TO END:
      it prefills the same prompt again and then decodes.

    A warmup pass makes those SAME two calls, untimed, and generates the same
    `n_new` tokens.  It has to: an engine's first full-length decode costs
    several times what every later one costs, so a warmup that generated a
    handful of tokens warmed the prefill and left the first timed repetition
    the first long decode of the process.  The median survived that, but the
    recorded spread was the cold pass rather than the run's own variation
    (zk-prover #106).

    The DECODE's own wall-clock time is the difference, which is the `n_new - 1`
    decode steps after the first generated position.

    Three rates come out of that, each over the positions IT covers, and the
    positions of the parts fall one short of the end-to-end count because the
    first generated position is produced inside the prefill and is charged to
    neither part:

      prefill      n_prompt / prefill_s
      decode       (n_new - 1) / decode_s
      end to end   (n_prompt + n_new) / wall_s

    None of the three is an "overhead" denominator, and no headline rate
    subtracts the prompt out any more: the paper quotes a cost per proved
    position and the reader applies their own mix of prompt and generated
    positions (the decision of 2026-09-08 in zk-results, `docs/decisions/
    2026-09-08-cost-per-proved-position-not-an-overhead-multiple.md`).

    None of the three is an end-to-end user wall-clock time either: model load
    and weight residency sit outside all of them, on both sides, as does the
    witness extraction on the proved side.
    """
    n_prompt = len(prompts[0])
    per_rep = []
    for _ in range(warmup):
        be.generate(prompts, 1)
        be.generate(prompts, n_new)
    be.sync()
    gen_sha = None
    for _ in range(reps):
        be.sync()
        t0 = time.perf_counter()
        be.generate(prompts, 1)
        be.sync()
        t1 = time.perf_counter()
        got = be.generate(prompts, n_new)
        be.sync()
        t2 = time.perf_counter()
        prefill_s, wall_s = t1 - t0, t2 - t1
        decode_s = wall_s - prefill_s
        if decode_s <= 0:
            raise RuntimeError(
                f"the {n_new}-token call took {wall_s:.4f}s and the prefill call "
                f"{prefill_s:.4f}s, so the decode's own time is not positive. The "
                "two calls are not measuring what this protocol says they are; no "
                "rate is reported from them.")
        gen_sha = _sha(got)
        per_rep.append({
            "prefill_s": prefill_s, "decode_s": decode_s, "wall_s": wall_s,
            "pos_s_prefill": n_prompt / prefill_s,
            "pos_s_decode": (n_new - 1) / decode_s,
            "pos_s_end_to_end": (n_prompt + n_new) / wall_s,
            "generated_sha256_16": gen_sha,
        })
    return per_rep, gen_sha
