"""B1: explicit, deterministic Qwen2 forward pass -- the computation to be attested.

Nothing from transformers is used at inference time, only the checkpoint weights:
every op is written out so the witness is exactly what a circuit would have to
prove, with no library fusion hiding an evaluation order.

Fixed evaluation order (all of it observable in `forward`):
  * tokens are processed with an explicit KV cache; a batched call over S tokens
    and S single-token calls produce the same values (checked in selftest.py);
  * RMSNorm: mean of squares (reduction) -> +eps -> rsqrt -> scale -> weight;
  * softmax: row max subtracted first, then exp, then the denominator reduction,
    then one reciprocal and a multiply (no division per element);
  * RoPE tables are precomputed in fp64 and rounded once to bf16, as preprocessed
    circuit constants.

The RMSNorm mean and the softmax denominator accumulate per `Spec.reduce`.  Matmul
accumulation is separate: the int8 matmuls accumulate integers, exact by
construction, while the two bf16 attention matmuls accumulate in fp32 and round
once -- a v1 gap B5 closes by quantizing them too (`Spec.qk_int8` is half of it).
"""
from dataclasses import dataclass, replace

import torch

from quant import (BF16, gemm_dtype, int_gemm, qscale_per_channel, qscale_per_token,
                   quantize, reduce_sum, store)


@dataclass
class Spec:
    """Which parts of the computation are quantized."""
    linear: str = "dynamic"   # fp32 | dynamic (per-token) | static (per-tensor, calibrated)
    nonlinear: str = "bf16"   # fp32 | bf16   -- dtype of every value between ops
    # accumulation for the two reductions, either fp32 or a correctly-rounded bf16
    # tree: fp32 | bf16 | bf16-norm (RMSNorm only) | bf16-softmax (denominator only)
    reduce: str = "fp32"
    qk_int8: bool = False     # probe: quantize the Q.K^T matmul as well
    fp32_logits: bool = False  # leave the final dequant unrounded (see README: NLL
                               # measures the logits themselves, so rounding them
                               # to bf16 charges the scheme for a sampling-invisible
                               # error -- this isolates that)
    # matmul kinds held out of int8 ("q","k","v","o","gate","up","down","lm_head"):
    # attributes the quantization loss to individual matrices
    fp32_matmuls: tuple = ()

    def asdict(self):
        return {"linear": self.linear, "nonlinear": self.nonlinear, "reduce": self.reduce,
                "qk_int8": self.qk_int8, "fp32_logits": self.fp32_logits,
                "fp32_matmuls": list(self.fp32_matmuls)}


class Recorder:
    """Collects one row per forward call for every witness column."""

    def __init__(self):
        self.cols = {}

    def add(self, key, t):
        self.cols.setdefault(key, []).append(t.detach().clone())

    def stack(self):
        return {k: torch.stack(v) if v[0].shape == v[-1].shape else v
                for k, v in self.cols.items()}


class Linear:
    """One weight matrix, plus its quantization.

    Weights are transposed to (n_in, n_out) so the matmul is x @ W and the
    per-output-channel scale is a reduction over dim 0.  Quantized weights are
    held in the float dtype their reduction width needs, so BLAS is still exact.
    """

    def __init__(self, w, bias, name, spec, static_scale=None):
        w = w.t().contiguous().float()
        self.name, self.spec = name, spec
        self.n_in, self.n_out = w.shape
        self.bias = None if bias is None else bias.clone().float()
        self.static_scale = static_scale
        if spec.linear == "fp32":
            self.w, self.qw, self.s_w = w, None, None
        else:
            self.s_w = qscale_per_channel(w)
            self.qw = quantize(w, self.s_w).to(gemm_dtype(self.n_in))
            self.w = None

    def __call__(self, x, m, store_out=True):
        if m.calib is not None:
            # per-token max|x|: its max over a calibration set is the static
            # per-tensor range, and its spread is what outliers.py reports.
            m.calib.setdefault(self.name, []).append(x.abs().amax(-1).clone())
        if self.qw is None:
            y = x @ self.w
        else:
            s_x = self.static_scale if self.spec.linear == "static" else qscale_per_token(x)
            qx = quantize(x, s_x)
            acc = int_gemm(qx, self.qw)
            m.note(self.name, acc)
            if m.rec is not None:
                m.rec.add(f"{self.name}/qx", qx.to(torch.int8))
                m.rec.add(f"{self.name}/acc", acc.to(torch.int32))
                m.rec.add(f"{self.name}/s_x", s_x.to(BF16).expand(x.shape[0], 1))
            y = acc * s_x * self.s_w
        if self.bias is not None:
            y = y + self.bias
        return m.store(y) if store_out else y


class Cache:
    """Per-layer KV cache; keys/values are appended in token order."""

    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.n = 0

    def append(self, i, k, v):
        self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], 0)
        self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], 0)
        return self.k[i], self.v[i]


class Model:
    def __init__(self, path, spec, static_scales=None, track_acc=False):
        from transformers import AutoModelForCausalLM

        # the checkpoint is bf16, so loading it as bf16 and widening one tensor at a
        # time in Linear is value-identical to loading fp32 and halves peak RAM
        hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        hf.eval()
        c = hf.config
        self.spec, self.cfg = spec, c
        self.d, self.L = c.hidden_size, c.num_hidden_layers
        self.nh, self.nkv = c.num_attention_heads, c.num_key_value_heads
        self.hd = c.hidden_size // c.num_attention_heads
        self.rep = self.nh // self.nkv
        self.rec, self.calib, self.acc_max = None, None, ({} if track_acc else None)

        ss = static_scales or {}
        self.emb = hf.model.embed_tokens.weight.data.clone()   # kept bf16
        self.layers = []
        for i in range(self.L):
            b = hf.model.layers[i]
            a, mlp = b.self_attn, b.mlp
            lin = lambda mod, n: Linear(mod.weight.data, None if mod.bias is None else mod.bias.data,
                                        f"L{i}.{n}", self.spec_for(n), ss.get(f"L{i}.{n}"))
            self.layers.append(dict(
                norm1=b.input_layernorm.weight.data.clone().float(),
                norm2=b.post_attention_layernorm.weight.data.clone().float(),
                q=lin(a.q_proj, "q"), k=lin(a.k_proj, "k"), v=lin(a.v_proj, "v"),
                o=lin(a.o_proj, "o"), gate=lin(mlp.gate_proj, "gate"),
                up=lin(mlp.up_proj, "up"), down=lin(mlp.down_proj, "down")))
            hf.model.layers[i] = torch.nn.Identity()   # release the checkpoint copy as we go
        self.final_norm = hf.model.norm.weight.data.clone().float()
        self.head = Linear(self.emb, None, "lm_head", self.spec_for("lm_head"), ss.get("lm_head"))
        del hf

        self.eps = self.store(torch.tensor(float(c.rms_norm_eps)))
        self.inv_d = self.store(torch.tensor(1.0 / self.d))
        self.attn_scale = self.hd ** -0.5       # exact power of two for hd=64
        theta = float(c.rope_parameters["rope_theta"])
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        self.inv_freq = inv

    # -- primitives -------------------------------------------------------
    def spec_for(self, kind):
        return replace(self.spec, linear="fp32") if kind in self.spec.fp32_matmuls else self.spec

    def store(self, x):
        return store(x, self.spec.nonlinear == "bf16")

    def note(self, name, acc):
        if self.acc_max is not None:
            self.acc_max[name] = max(self.acc_max.get(name, 0.0), acc.abs().amax().item())

    def unary(self, key, x, fn):
        y = self.store(fn(x))
        if self.rec is not None:
            self.rec.add(f"{key}/in", x.to(BF16))
            self.rec.add(f"{key}/out", y.to(BF16))
        return y

    def red(self, which):
        return "bf16" if self.spec.reduce in ("bf16", f"bf16-{which}") else "fp32"

    def rmsnorm(self, x, w, key):
        ms = self.store(reduce_sum(self.store(x * x), -1, True, self.red("norm")) * self.inv_d)
        r = self.unary(f"{key}.rsqrt", self.store(ms + self.eps), torch.rsqrt)
        return self.store(self.store(x * r) * w)

    def rope(self, pos):
        ang = pos.double().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        ang = torch.cat([ang, ang], dim=-1)
        return self.store(ang.cos().float()), self.store(ang.sin().float())

    def rotate(self, t, cos, sin):
        h = self.hd // 2
        r = torch.cat([-t[..., h:], t[..., :h]], dim=-1)
        return self.store(self.store(t * cos) + self.store(r * sin))

    def attention(self, q, k, v, pos, li):
        s, c = q.shape[0], k.shape[0]
        q = q.transpose(0, 1)                                    # (nh, S, hd)
        k = k.permute(1, 0, 2).repeat_interleave(self.rep, 0)    # (nh, C, hd)
        v = v.permute(1, 0, 2).repeat_interleave(self.rep, 0)
        if self.spec.qk_int8:
            s_q, s_k = qscale_per_token(q), qscale_per_token(k)
            acc = int_gemm(quantize(q, s_q), quantize(k, s_k).transpose(1, 2))
            self.note(f"L{li}.qk", acc)
            sc = self.store(acc * s_q * s_k.transpose(1, 2))
        else:
            sc = self.store(q @ k.transpose(1, 2))
        sc = self.store(sc * self.attn_scale)
        mask = torch.arange(c).unsqueeze(0) > pos.unsqueeze(1)   # causal, absolute positions
        sc = sc.masked_fill(mask.unsqueeze(0), float("-inf"))
        if self.rec is not None:
            self.rec.add(f"L{li}.scores", sc.to(BF16))
        e = self.unary(f"L{li}.exp", self.store(sc - sc.amax(-1, keepdim=True)), torch.exp)
        den = self.store(reduce_sum(e, -1, True, self.red("softmax")))
        p = self.store(e * self.unary(f"L{li}.recip", den, torch.reciprocal))
        if self.rec is not None:
            self.rec.add(f"L{li}.probs", p.to(BF16))
        out = self.store(p @ v)
        return out.transpose(0, 1).reshape(s, self.nh * self.hd)

    # -- forward ----------------------------------------------------------
    def forward(self, ids, cache=None):
        cache = cache or Cache(self.L)
        pos = torch.arange(cache.n, cache.n + len(ids))
        cos, sin = self.rope(pos)
        x = self.emb[ids].float()
        for i, l in enumerate(self.layers):
            h = self.rmsnorm(x, l["norm1"], f"L{i}.norm1")
            q = l["q"](h, self).view(-1, self.nh, self.hd)
            k = l["k"](h, self).view(-1, self.nkv, self.hd)
            v = l["v"](h, self).view(-1, self.nkv, self.hd)
            q = self.rotate(q, cos.unsqueeze(1), sin.unsqueeze(1))
            k = self.rotate(k, cos.unsqueeze(1), sin.unsqueeze(1))
            kk, vv = cache.append(i, k, v)
            x = self.store(x + l["o"](self.attention(q, kk, vv, pos, i), self))
            h = self.rmsnorm(x, l["norm2"], f"L{i}.norm2")
            g = self.unary(f"L{i}.silu", l["gate"](h, self), torch.nn.functional.silu)
            x = self.store(x + l["down"](self.store(g * l["up"](h, self)), self))
            if self.rec is not None:
                self.rec.add(f"L{i}.x", x.to(BF16))     # committed residual stream
        cache.n += len(ids)
        h = self.rmsnorm(x, self.final_norm, "final_norm")
        return self.head(h, self, store_out=not self.spec.fp32_logits), cache
