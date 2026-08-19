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

from quant import (BF16, gemm_dtype, int_gemm, int_gemm_exact, qscale,
                   qscale_per_channel, qscale_per_token, quantize, reduce_sum,
                   round_hafz, store)


@dataclass
class Spec:
    """Which parts of the computation are quantized."""
    linear: str = "dynamic"   # fp32 | dynamic (per-token) | static (per-tensor, calibrated)
    nonlinear: str = "bf16"   # fp32 | bf16   -- dtype of every value between ops
    # accumulation for the two reductions, either fp32 or a correctly-rounded bf16
    # tree: fp32 | bf16 | bf16-norm (RMSNorm only) | bf16-softmax (denominator only)
    reduce: str = "fp32"
    qk_int8: bool = False     # probe: quantize the Q.K^T matmul as well
    # B5 probes.  qk_bits is (q bits, k bits) for the quantized Q.K^T path -- the
    # M31 window is a per-matmul budget and Q.K^T reduces over hd, not dff, so
    # int12 x int11 costs the same committed cell as int8 x int8.
    qk_bits: tuple = (8, 8)
    # the integer softmax: integer scores -> integer max shift -> one table whose
    # output is already a fixed-point numerator -> integer A.V, with the division
    # deferred into the output dequant.  Needs K and V scales that are constant
    # along the context axis, i.e. calibrated per (layer, head) and per
    # (layer, head, channel) -- see b5/README.md.
    int_softmax: bool = False
    num_bits: int = 8         # numerator width of the integer softmax
    # "static": K/V scales calibrated offline, the only schedule-invariant choice;
    # "cache": recomputed from the current KV cache every step -- a quality
    # reference, not a candidate, because it re-quantizes committed columns
    attn_scale_mode: str = "static"
    # "token": a dynamic Q scale per query, so the exp-table argument carries a
    # per-token exponent; "head": one calibrated Q scale per (layer, head), which
    # makes that exponent a public constant and lets a single integer-indexed exp
    # table serve every head -- the difference between 4 and 5 cells per score
    q_scale_mode: str = "token"
    # "": none; "all": every quantization scale rounded up to a power of two;
    # "attn": only the Q/K/V scales, which is what makes the exp-table input a bit
    # window of the score instead of a mantissa product
    pow2_scales: str = ""
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
                "fp32_matmuls": list(self.fp32_matmuls),
                "qk_bits": list(self.qk_bits), "int_softmax": self.int_softmax,
                "num_bits": self.num_bits, "pow2_scales": self.pow2_scales,
                "attn_scale_mode": self.attn_scale_mode,
                "q_scale_mode": self.q_scale_mode}


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
            self.s_w = qscale_per_channel(w, spec.pow2_scales == "all")
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
            s_x = (self.static_scale if self.spec.linear == "static"
                   else qscale_per_token(x, self.spec.pow2_scales == "all"))
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
    def __init__(self, path, spec, static_scales=None, track_acc=False, attn_scales=None):
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
        # B5: per-(layer, head) max|k| and per-(layer, head, channel) max|v|, the
        # two things the integer softmax needs to be constant along the context
        # axis.  Set to {} to record them (ppl.py's calibration pass).
        self.attn_calib = None
        self.attn_scales = attn_scales
        self.clamped = None

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

    def attention_int(self, q, k, v, pos, li):
        """The integer softmax (plan, B5 constraints from the B4 review).

        Every step on the probs multiplicity rung is an integer op or a table
        lookup; the only rounding left is one per *output element*:

          integer scores  S = qQ qK^T                      (Freivalds, exact)
          integer shift   d = max_c S - S >= 0             (a range check)
          one table       n = round(NMAX exp(-d s_q s_k a))  fixed-point numerator
          integer sum     Z = sum_c n                      (exact, the denominator)
          integer A.V     O = n qV                         (Freivalds, exact)
          out = O * s_v / Z                                one rounding per element

        `s_k` and `s_v` must not vary along the context axis or neither the max
        shift nor the A.V accumulator is an integer statement: `s_k` is one scalar
        per (layer, head) and `s_v` one per (layer, head, channel), both fixed
        offline.  A per-channel V scale factors out of a reduction over keys the
        same way a per-output-channel weight scale factors out of a matmul.
        """
        nh, ss, cc = q.shape[0], q.shape[1], k.shape[1]
        qb, kb = self.spec.qk_bits
        qmax_q, qmax_k = (1 << (qb - 1)) - 1, (1 << (kb - 1)) - 1
        nmax = (1 << self.spec.num_bits) - 1
        p2 = self.spec.pow2_scales in ("all", "attn")
        s_k, s_v, s_qh = self.kv_scales(li, k, v)                # (nh,1,1), (nh,1,hd)
        # a calibrated per-head Q scale when the spec asks for one, so the
        # exp-table argument's exponent is a public constant; dynamic otherwise
        s_q = qscale(q, -1, qmax_q, p2) if s_qh is None else s_qh
        qq = quantize(q, s_q, qmax_q)
        qk = quantize(k, s_k, qmax_k)
        qv = quantize(v, s_v)
        if self.clamped is not None:
            self.tally("k", k / s_k, qmax_k)
            self.tally("v", v / s_v, 127)

        bound = qmax_q * qmax_k * self.hd
        sc = int_gemm_exact(qq, qk.transpose(1, 2), bound)       # exact integers
        self.note(f"L{li}.qk", sc)
        if self.rec is not None:
            self.rec.add(f"L{li}.sqk", (s_q * s_k * self.attn_scale).expand(
                q.shape[0], q.shape[1] if s_q.shape[1] > 1 else 1, 1).float())
            self.rec.add(f"L{li}.qq", qq.to(torch.int16))
            self.rec.add(f"L{li}.kk", qk.to(torch.int16))
            self.rec.add(f"L{li}.vv", qv.to(torch.int8))
            self.rec.add(f"L{li}.qkacc", sc.to(torch.int32))
        mask = torch.arange(cc).unsqueeze(0) > pos.unsqueeze(1)  # causal
        sc = sc.masked_fill(mask.unsqueeze(0), -float(bound + 1))
        m = sc.amax(-1, keepdim=True)                            # integer row max
        d = m - sc                                               # >= 0, integer
        # the one dynamic-scale product left on this rung: B5 commits its result
        # and proves nothing about it (v1 gap).  With power-of-two scales it is a
        # bit window of `d` and needs no gadget at all.
        x = self.store((-d * (s_q * s_k * self.attn_scale).to(d.dtype)).float())
        n = round_hafz(nmax * torch.exp(x.double())).clamp_(0, nmax).float()
        n = n.masked_fill(mask.unsqueeze(0), 0.0)
        if self.rec is not None:
            self.rec.add(f"L{li}.expnum/in", x.to(BF16))
            self.rec.add(f"L{li}.expnum/out", n.to(torch.int32))
        # fp32 stops being exact on integers above 2^24, which a 16-bit numerator
        # reaches at C = 256; the sum is the denominator, so it has to stay exact.
        z = n.double().sum(-1, keepdim=True)                     # exact integer
        o = int_gemm_exact(n, qv, nmax * 127 * cc)               # exact integer
        self.note(f"L{li}.av", o)
        rz = self.unary(f"L{li}.recip", self.store(z), torch.reciprocal)
        out = self.store(o * s_v * rz)
        return out.transpose(0, 1).reshape(ss, self.nh * self.hd)

    def kv_scales(self, li, k, v):
        """Context-axis-constant K and V scales, calibrated offline.

        Without `attn_scales` they fall back to the current cache's own maxima,
        which is *not* schedule-invariant (the scale changes as the cache grows)
        and is reported separately as a quality reference, never as a candidate.
        """
        p2 = self.spec.pow2_scales in ("all", "attn")
        qmax_k = (1 << (self.spec.qk_bits[1] - 1)) - 1
        if self.attn_scales is None or self.spec.attn_scale_mode == "cache":
            return (qscale(k.reshape(k.shape[0], -1), -1, qmax_k, p2).unsqueeze(-1),
                    qscale(v, 1, 127, p2), None)
        s_k, s_v = self.attn_scales[f"L{li}.k"], self.attn_scales[f"L{li}.v"]
        s_q = self.attn_scales.get(f"L{li}.q") if self.spec.q_scale_mode == "head" else None
        return s_k, s_v, s_q

    def tally(self, kind, scaled, qmax):
        t = self.clamped.setdefault(kind, [0, 0])
        t[0] += int((scaled.abs() > qmax + 0.5).sum())
        t[1] += scaled.numel()

    def attention(self, q, k, v, pos, li):
        s, c = q.shape[0], k.shape[0]
        q = q.transpose(0, 1)                                    # (nh, S, hd)
        k = k.permute(1, 0, 2).repeat_interleave(self.rep, 0)    # (nh, C, hd)
        v = v.permute(1, 0, 2).repeat_interleave(self.rep, 0)
        if self.rec is not None:
            # the RoPE outputs, i.e. exactly the two operands Q.K^T contracts:
            # B5's order probe needs them to show which accumulation is exact
            self.rec.add(f"L{li}.qrope", q.to(BF16))
            self.rec.add(f"L{li}.krope", k.to(BF16))
        if self.attn_calib is not None:
            # the ranges the integer softmax has to fix offline: K and V along the
            # context axis, and (for a public exp-table exponent) Q as well
            self.attn_calib.setdefault(f"L{li}.k", []).append(k.abs().amax((1, 2)))
            self.attn_calib.setdefault(f"L{li}.v", []).append(v.abs().amax(1))
            self.attn_calib.setdefault(f"L{li}.q", []).append(q.abs().amax((1, 2)))
        if self.spec.int_softmax:
            return self.attention_int(q, k, v, pos, li)
        if self.spec.qk_int8:
            qb, kb = self.spec.qk_bits
            qmax_q, qmax_k = (1 << (qb - 1)) - 1, (1 << (kb - 1)) - 1
            p2 = self.spec.pow2_scales in ("all", "attn")
            s_q = qscale(q, -1, qmax_q, p2)
            s_k = qscale(k, -1, qmax_k, p2)
            acc = int_gemm_exact(quantize(q, s_q, qmax_q),
                                 quantize(k, s_k, qmax_k).transpose(1, 2),
                                 qmax_q * qmax_k * self.hd)
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
