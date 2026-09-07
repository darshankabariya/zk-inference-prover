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
import math
import os
from dataclasses import dataclass, replace

import torch

from .quant import (BF16, INT8_MAX, INT_MM_MIN_ROWS, bf16_away, int_gemm,
                   int_gemm_exact, qscale, qscale_per_channel, qscale_per_token,
                   quantize, reduce_sum, round_hafz, store, weight_dtype)

# The Gemma-3 text tower, the only family here that is not Qwen2/Llama shaped.
# Everything it needs beyond the shared path is a *branch* on the checkpoint's
# own config, never an edit to the arithmetic the frozen spec's numbers were
# measured with (briefs/e5a-gemma.md):
#   * the embedding is multiplied by sqrt(hidden_size) before the first layer;
#   * every RMSNorm gain is `1 + w`, not `w`;
#   * Q and K are RMSNormed per head before RoPE (QK-norm);
#   * four norms per layer -- the two extra ones sit *after* the attention and
#     after the MLP, before the residual add;
#   * layers alternate local (sliding-window) and global attention, each type
#     with its own RoPE theta;
#   * the activation is GELU-tanh, not SiLU, and the logits may be softcapped.
GEMMA3 = "gemma3_text"

# The Gemma-4 text tower.  Same lineage, different family: it keeps Gemma-3's
# frame (the embedding normalizer, four norms per layer, QK-norm, the 5:1
# local/global alternation with a RoPE table per type, GELU-tanh, the logit
# softcap) and changes six things, every one of them again a branch on the
# checkpoint's own config (briefs/e5c-gemma4.md):
#   * the head geometry is per *layer type* -- `head_dim` 256 with 16 KV heads on
#     the local layers, 512 with 4 on the global ones, so `q_proj`'s own output
#     width differs by layer.  These are heterogeneous config attributes and
#     reading them off the top-level config raises;
#   * the global RoPE is `proportional` with `partial_rotary_factor` 0.25: only
#     the first quarter of the angle pairs get a frequency, the rest are NoPE;
#   * `attention_k_eq_v` -- a global layer has no `v_proj` at all.  V is
#     `k_proj(x)` taken *before* K's norm, and it is never rotated;
#   * V goes through a **scale-free** RMSNorm on every layer type;
#   * the RMSNorm gain is `w`, not `1 + w` (Gemma3n-style), and the attention
#     scale is exactly 1.0 -- there is no `query_pre_attn_scalar` at all;
#   * each layer's whole output is multiplied by a per-layer `layer_scalar`.
# Per-layer embeddings, KV-shared layers, the double-wide MLP and the MoE block
# are in the class but off in the 31B checkpoint; each is refused, not guessed.
GEMMA4 = "gemma4_text"


@dataclass
class Spec:
    """Which parts of the computation are quantized."""
    # fp32 | dynamic (per-token) | static (per-tensor, calibrated)
    # | w8a16 (per-channel int8 weights, float activations -- B9's comparator)
    linear: str = "dynamic"
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
    # window of the score instead of a mantissa product; "linear": only the weight
    # and activation scales of the matmuls, which is what makes the *dequant*
    # `bf16(acc s_x s_w)` a single rounding plus an exponent add instead of a chain
    # of three roundings through two mantissa-product tables (B6's ladder)
    pow2_scales: str = ""
    # How the dequant `acc s_x s_w -> bf16` is evaluated.  "fp32" is B1's original
    # contract: accumulate the product in fp32 and round once at the end.
    # "bf16-chain" is what a gadget can actually prove when the scales are general
    # bf16 values -- bf16(bf16(bf16(acc) s_x) s_w), three correctly-rounded steps --
    # so the difference between the two is the numeric price of the general-scale
    # dequant.  With power-of-two scales the two coincide exactly.
    dequant: str = "fp32"
    # The chain's bf16 rounding mode: "even" is B1's contract (`.to(bfloat16)`);
    # "away" (ties away from zero) needs no sticky bit -- one fewer witness column
    # and half the table rows in B6's gadget (25.12 against 26.12 cells) -- but had
    # no perplexity number until the B6 review (b6/README, open gap 5).
    dequant_round: str = "even"
    # The attention output feeds `o_proj`, which quantizes it to int8 anyway, so
    # `s_v/(Z s_o)` can be folded into a *single* rounding straight to int8 -- which
    # deletes the bf16 waypoint, the reciprocal lookup and the requantize at every
    # output element (B5 review; B6 prices the gadget at 9 cells against 15 + 4 + 15).
    attn_out_int8: bool = False
    # B6 review: the int8 gadget's no-wrap argument needs `den = Z`, i.e. a
    # power-of-two output scale -- a general bf16 `s_o` puts its mantissa into the
    # denominator and `q den` outgrows p at C = 16K.  B6's probes measured a bf16
    # `s_o`; this knob makes the output scale the one the gadget can prove.
    attn_out_pow2: bool = False
    # power-of-two weight and activation scales for *these* matmul kinds only,
    # whatever `pow2_scales` says: attributes `pow2-dequant`'s +2.29% to the
    # matmul groups it comes from (B6 review)
    pow2_matmuls: tuple = ()
    fp32_logits: bool = False  # leave the final dequant unrounded (see README: NLL
                               # measures the logits themselves, so rounding them
                               # to bf16 charges the scheme for a sampling-invisible
                               # error -- this isolates that)
    # matmul kinds held out of int8 ("q","k","v","o","gate","up","down","lm_head"):
    # attributes the quantization loss to individual matrices
    fp32_matmuls: tuple = ()
    # --- stream Y: requantization probes ------------------------------------
    # Width of the attention-output requant.  `attn_out_int8` folds the softmax
    # division, the value scale and o's requantize into a single rounding; this
    # widens that rounding's target from int8 to `attn_out_bits`, so `o_proj`
    # consumes the A.V product at a wider integer width -- the delayed form of
    # the requant.  8 is exactly the b10-final path.
    attn_out_bits: int = 8
    # Matmul kinds whose output the consumer requantizes with nothing genuinely
    # nonlinear between: their dequant chain is delayed away, i.e. the output is
    # carried as the exact product `acc * s_x * s_w` (fp64, no bf16 waypoint,
    # no rounding) and the consumer's own quantize becomes the single rounding
    # -- B6's `int8` dequant variant is the gadget this models.  "v" (only the
    # KV cache between the dequant and the A.V quantize) and "up" (the SwiGLU
    # product is linear in up's operand) are the sites; the product `g * up`
    # also stays unrounded when "up" is delayed.
    delay_matmuls: tuple = ()

    def asdict(self):
        return {"linear": self.linear, "nonlinear": self.nonlinear, "reduce": self.reduce,
                "qk_int8": self.qk_int8, "fp32_logits": self.fp32_logits,
                "fp32_matmuls": list(self.fp32_matmuls),
                "qk_bits": list(self.qk_bits), "int_softmax": self.int_softmax,
                "num_bits": self.num_bits, "pow2_scales": self.pow2_scales,
                "attn_scale_mode": self.attn_scale_mode,
                "q_scale_mode": self.q_scale_mode,
                "dequant": self.dequant, "dequant_round": self.dequant_round,
                "attn_out_int8": self.attn_out_int8,
                "attn_out_pow2": self.attn_out_pow2,
                "pow2_matmuls": list(self.pow2_matmuls),
                "attn_out_bits": self.attn_out_bits,
                "delay_matmuls": list(self.delay_matmuls)}


class Recorder:
    """Collects one row per forward call for every witness column."""

    def __init__(self):
        self.cols = {}

    def add(self, key, t):
        self.cols.setdefault(key, []).append(t.detach().cpu().clone())

    def stack(self):
        return {k: torch.stack(v) if v[0].shape == v[-1].shape else v
                for k, v in self.cols.items()}


class Linear:
    """One weight matrix, plus its quantization.

    Weights are transposed to (n_in, n_out) so the matmul is x @ W and the
    per-output-channel scale is a reduction over dim 0.  Quantized weights are
    held in the float dtype their reduction width needs, so BLAS is still exact.
    """

    def __init__(self, w, bias, name, spec, static_scale=None, device="cpu"):
        w = w.to(device).t().contiguous().float()
        self.name, self.spec = name, spec
        # the matmul kind ("q", "v", "up", ...) -- stream Y's delayed
        # requantization is a per-kind knob
        self.delayed = name.rsplit(".", 1)[-1] in spec.delay_matmuls
        self.n_in, self.n_out = w.shape
        self.bias = None if bias is None else bias.to(device).clone().float()
        self.static_scale = None if static_scale is None else static_scale.to(device)
        self.p2 = spec.pow2_scales in ("all", "linear")
        if spec.linear == "fp32":
            self.w, self.qw, self.s_w = w, None, None
        elif spec.linear == "w8a16":
            # B9's comparator: per-channel int8 weights, float activations.  The
            # weight is quantized and dequantized once, at load, so the matmul is
            # the float one and no activation is ever quantized -- weight-only
            # int8, the strong baseline against which "beats a public INT8
            # variant" has to be read.
            self.s_w = qscale_per_channel(w, self.p2)
            self.w = quantize(w, self.s_w) * self.s_w
            self.qw = None
        else:
            self.s_w = qscale_per_channel(w, self.p2)
            self.qw = quantize(w, self.s_w).to(weight_dtype(self.n_in, device))
            self.w = None

    def __call__(self, x, m, store_out=True, pre=None):
        if m.calib is not None and x is not None:
            # per-token max|x|: its max over a calibration set is the static
            # per-tensor range, and its spread is what outliers.py reports.
            m.calib.setdefault(self.name, []).append(x.abs().amax(-1).clone())
        if self.qw is None:
            y = x @ self.w
        elif pre is not None:
            # the activation was quantized by its producer (the attention output's
            # single rounding straight to int8), so there is nothing to requantize
            qx, s_x = pre
            qmax_pre = (1 << (self.spec.attn_out_bits - 1)) - 1
            if qmax_pre > INT8_MAX:
                # a widened attention output: `torch._int_mm` clips its operand
                # to int8, so the wider operand takes the exact fp64 path
                acc = int_gemm_exact(qx, self.qw, qmax_pre * INT8_MAX * self.n_in)
            else:
                acc = int_gemm(qx, self.qw)
            m.note(self.name, acc)
            if m.rec is not None:
                m.rec.add(f"{self.name}/qx",
                          qx.to(torch.int8 if qmax_pre <= INT8_MAX else torch.int16))
                m.rec.add(f"{self.name}/acc", acc.to(torch.int32))
                m.rec.add(f"{self.name}/s_x", s_x.to(BF16).expand(qx.shape[0], 1))
            y = m.dequant(acc, s_x, self.s_w)
        else:
            s_x = (self.static_scale if self.spec.linear == "static"
                   else qscale_per_token(x, self.p2))
            qx = quantize(x, s_x)
            if self.delayed:
                # stream Y: the consumer requantizes this output, so the dequant
                # chain is delayed away.  The accumulator is taken exactly (the
                # fp32 rounding `int_gemm` ends with would perturb it above 2^24)
                # and the output is the exact product -- no bf16 waypoint; the
                # consumer's quantize is the one rounding left.
                if self.qw.dtype == torch.int8:
                    rows = qx.shape[0]
                    a = qx.to(torch.int8)
                    if rows < INT_MM_MIN_ROWS:
                        a = torch.nn.functional.pad(a, (0, 0, 0, INT_MM_MIN_ROWS - rows))
                    acc = torch._int_mm(a, self.qw)[:rows].double()
                else:
                    acc = (qx.to(self.qw.dtype) @ self.qw).double()
            else:
                acc = int_gemm(qx, self.qw)
            m.note(self.name, acc)
            if m.rec is not None:
                m.rec.add(f"{self.name}/x", x.to(BF16))
                m.rec.add(f"{self.name}/qx", qx.to(torch.int8))
                m.rec.add(f"{self.name}/acc", acc.to(torch.int32))
                m.rec.add(f"{self.name}/s_x", s_x.to(BF16).expand(x.shape[0], 1))
            y = (acc * s_x * self.s_w if self.delayed
                 else m.dequant(acc, s_x, self.s_w))
        if self.bias is not None:
            y = y + self.bias
        return m.store(y) if store_out and not self.delayed else y


class Cache:
    """Per-layer KV cache; keys/values are appended in token order.

    B8: a cache may hold several sequences at once (packed decode).  Every cached
    key carries its sequence id and position; a query attends to a key iff they
    share the sequence and the key's position is not past the query's.  With one
    sequence that is exactly the causal mask, and since every other op in the
    forward pass is per token, packing B sequences is bit-identical to running
    them one at a time -- for the integer attention, masked keys contribute 0 to
    the numerator sum and to A.V.
    """

    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.n = 0
        self.pos = None     # (C,) position of every cached key within its sequence
        self.seq = None     # (C,) sequence id of every cached key

    def append(self, i, k, v):
        self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], 0)
        self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], 0)
        return self.k[i], self.v[i]

    def append_meta(self, pos, seq):
        self.pos = pos if self.pos is None else torch.cat([self.pos, pos])
        self.seq = seq if self.seq is None else torch.cat([self.seq, seq])
        self.n += len(pos)

    def keep(self, seq_ids):
        """Drop every key whose sequence is not in `seq_ids` (finished sequences)."""
        m = torch.isin(self.seq, torch.as_tensor(list(seq_ids), device=self.seq.device))
        self.k = [None if t is None else t[m] for t in self.k]
        self.v = [None if t is None else t[m] for t in self.v]
        self.pos, self.seq = self.pos[m], self.seq[m]
        self.n = len(self.pos)

    def merge(self, other, seq_id):
        """Take a single-sequence cache (a prompt prefilled on its own) in as `seq_id`."""
        for i in range(len(self.k)):
            self.append(i, other.k[i], other.v[i])
        self.append_meta(other.pos, torch.full_like(other.pos, seq_id))


def llama3_inv_freq(inv, rp):
    """Llama-3's frequency scaling (transformers' `_compute_llama3_parameters`),
    evaluated in fp64 like the rest of the RoPE constants."""
    factor, lo, hi = float(rp["factor"]), float(rp["low_freq_factor"]), float(rp["high_freq_factor"])
    old = float(rp["original_max_position_embeddings"])
    wavelen = 2 * math.pi / inv
    scaled = torch.where(wavelen > old / lo, inv / factor, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    mid = (wavelen >= old / hi) & (wavelen <= old / lo)
    return torch.where(mid, (1 - smooth) * scaled / factor + smooth * scaled, scaled)


def proportional_inv_freq(rp, hd, theta):
    """Gemma-4's global RoPE (`_compute_proportional_rope_parameters`), in fp64.

    Only the first `partial_rotary_factor * head_dim / 2` angle pairs get a
    frequency; the remaining pairs are NoPE -- `inv_freq` exactly 0, so they still
    go through the rotation algebra (cos 1, sin 0) but carry no position.  At the
    31B's global layers that is 64 rotated pairs and 192 NoPE ones out of 256.
    """
    pairs = int(float(rp.get("partial_rotary_factor", 1.0)) * hd // 2)
    rot = 1.0 / (theta ** (torch.arange(0, 2 * pairs, 2, dtype=torch.float64) / hd))
    nope = torch.zeros(hd // 2 - pairs, dtype=torch.float64)
    return torch.cat([rot, nope]) / float(rp.get("factor", 1.0))


def is_pow2(x):
    """Is the float `x` an exact power of two?"""
    return math.frexp(x)[0] == 0.5


class Model:
    def __init__(self, path, spec, static_scales=None, track_acc=False, attn_scales=None,
                 device=None):
        from transformers import AutoModelForCausalLM

        device = device or os.environ.get("ZK_DEVICE", "cpu")
        # the checkpoint is bf16, so loading it as bf16 and widening one tensor at a
        # time in Linear is value-identical to loading fp32 and halves peak RAM
        hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        hf.eval()
        c = hf.config
        # a multimodal checkpoint keeps the language model one level down
        c = getattr(c, "text_config", None) or c
        base = getattr(hf.model, "language_model", hf.model)
        family = str(getattr(c, "model_type", ""))
        self.gemma4 = family == GEMMA4
        self.gemma = family in (GEMMA3, GEMMA4)
        if family.startswith("gemma") and not self.gemma:
            raise NotImplementedError(
                f"model_type {family}: only {GEMMA3} and {GEMMA4} are implemented -- "
                f"see briefs/e5a-gemma.md and briefs/e5c-gemma4.md")
        if self.gemma:
            if getattr(c, "attn_logit_softcapping", None):
                raise NotImplementedError("attn_logit_softcapping")
            # Gemma-4 declares `use_bidirectional_attention: "vision"`, i.e.
            # bidirectional over image spans only, so a text-only eval stays
            # causal; "all" would need a different mask entirely.
            bidi = getattr(c, "use_bidirectional_attention", None)
            if bidi not in (None, False, "vision"):
                raise NotImplementedError(f"use_bidirectional_attention {bidi}")
        if self.gemma4:
            # Present in `Gemma4TextConfig` and off in the 31B checkpoint.  Each is
            # a real architectural feature this forward pass does not implement, so
            # each is refused rather than silently ignored on some other size.
            for flag in ("hidden_size_per_layer_input", "num_kv_shared_layers",
                         "use_double_wide_mlp", "enable_moe_block", "num_experts"):
                if getattr(c, flag, None):
                    raise NotImplementedError(f"{flag}={getattr(c, flag)}")
        self.spec, self.cfg, self.device = spec, c, device
        self.d, self.L = c.hidden_size, c.num_hidden_layers
        self.nh = c.num_attention_heads
        if self.gemma4:
            # `head_dim` and `num_key_value_heads` are per-layer-type here, and
            # reading either off the top-level config raises
            # (`AmbiguousGlobalPerLayerAttributeError`, a RuntimeError -- a
            # `getattr` default does not catch it).  One record per type; the
            # forward pass makes one of them current per layer.
            self.geoms = {t: self.geometry_of(c.per_layer_config[t])
                          for t in set(c.layer_types)}
            self.use_geometry(c.layer_types[0])
        else:
            self.nkv = c.num_key_value_heads
            self.hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
            self.rep = self.nh // self.nkv
        self.rec, self.calib, self.acc_max = None, None, ({} if track_acc else None)
        # B5: per-(layer, head) max|k| and per-(layer, head, channel) max|v|, the
        # two things the integer softmax needs to be constant along the context
        # axis.  Set to {} to record them (ppl.py's calibration pass).
        self.attn_calib = None
        self.attn_scales = (None if attn_scales is None else
                            {k: v.to(device) for k, v in attn_scales.items()})
        self.clamped = None

        # 1/sqrt(hd) is an exact power of two for hd = 64 (Qwen2.5-0.5B).  For
        # hd = 128 (Llama-3) it is 2^-3.5, and the integer softmax's exp-table
        # argument is a bit window of the score only while `s_q s_k a` is a power
        # of two -- so the scale is folded into W_q before quantization, as a
        # deployment would carry it, and the attention scale becomes 1.
        self.qn_fold = None        # Gemma only: folded into the Q-norm gain instead
        if self.gemma4:
            # There is no `query_pre_attn_scalar` in a Gemma-4 config and no
            # 1/sqrt(head_dim) either: HF's `Gemma4TextAttention.scaling` is
            # literally 1.0, the QK-norms absorb it.  Nothing to fold, so B5's
            # exp-table argument is a bit window of the score for free.
            self.q_fold, self.attn_scale = None, 1.0
        elif self.gemma:
            # Gemma divides by sqrt(query_pre_attn_scalar), which is not always the
            # head dim (27B uses 168 against hd = 128).  That scale cannot be folded
            # into W_q the way it is below -- Q-norm is scale-invariant and would
            # undo it -- so it goes into the Q-norm gain, the first constant after
            # the normalization, whenever it is not already a power of two.
            a = float(getattr(c, "query_pre_attn_scalar", None) or self.hd) ** -0.5
            self.qn_fold = None if is_pow2(a) else a
            self.q_fold, self.attn_scale = None, (1.0 if self.qn_fold else a)
        else:
            log2_hd = self.hd.bit_length() - 1
            assert 1 << log2_hd == self.hd, self.hd
            self.q_fold = self.hd ** -0.5 if log2_hd % 2 else None
            self.attn_scale = 1.0 if self.q_fold else self.hd ** -0.5

        # the MLP's gate activation: SiLU (Qwen2/Llama) or GELU-tanh (Gemma)
        act = str(getattr(c, "hidden_activation", None) or "silu")
        if self.gemma and act != "gelu_pytorch_tanh":
            raise NotImplementedError(f"hidden_activation {act}")
        self.act = ((lambda t: torch.nn.functional.gelu(t, approximate="tanh"))
                    if self.gemma else torch.nn.functional.silu)

        ss = static_scales or {}
        self.emb = base.embed_tokens.weight.data.to(device).clone()   # kept bf16
        # Gemma's normalizer on the embedding lookup.  sqrt(d) is held at fp32 --
        # a model constant, exactly the value HF forms -- and the product is stored.
        self.embed_scale = float(self.d) ** 0.5 if self.gemma else None
        # Gemma-4's V-norm is an RMSNorm with `with_scale=False`: no learned gain at
        # all.  A gain of exactly 1.0 is that norm, and multiplying by it is exact
        # in both fp32 and bf16, so the shared `rmsnorm` needs no special case.
        self.no_gain = self.store(torch.tensor(1.0, device=device))
        self.layers = []
        for i in range(self.L):
            b = base.layers[i]
            a, mlp = b.self_attn, b.mlp
            lin = lambda w, mod, n: Linear(w, None if mod.bias is None else mod.bias.data,
                                           f"L{i}.{n}", self.spec_for(n), ss.get(f"L{i}.{n}"),
                                           device)
            # Gemma's RMSNorm gain is `1 + w`, kept at fp32: it is a property of the
            # model, not of the quantization, and bf16 would round it away.
            g1 = lambda mod: mod.weight.data.to(device).float() + 1.0
            qw = a.q_proj.weight.data
            if self.q_fold:
                qw = qw.to(device).float() * self.q_fold
            ld = dict(
                norm1=b.input_layernorm.weight.data.to(device).clone().float(),
                norm2=b.post_attention_layernorm.weight.data.to(device).clone().float(),
                q=lin(qw, a.q_proj, "q"), k=lin(a.k_proj.weight.data, a.k_proj, "k"),
                # `attention_k_eq_v`: a Gemma-4 global layer has no `v_proj` at all
                # and V comes off `k_proj`.  Every other checkpoint has one, so this
                # guard changes nothing that already loads.
                v=(lin(a.v_proj.weight.data, a.v_proj, "v")
                   if getattr(a, "v_proj", None) is not None else None),
                o=lin(a.o_proj.weight.data, a.o_proj, "o"),
                gate=lin(mlp.gate_proj.weight.data, mlp.gate_proj, "gate"),
                up=lin(mlp.up_proj.weight.data, mlp.up_proj, "up"),
                down=lin(mlp.down_proj.weight.data, mlp.down_proj, "down"))
            if self.gemma4:
                # Gemma-4's gain is `w`, not `1 + w`, so `norm1`/`norm2` are already
                # right as loaded.  `layer_scalar` is held at fp32 for the same
                # reason the norm gain and sqrt(d) are: it is a property of the
                # model, and rounding it into the value contract would be evaluating
                # a different one.  Its product with the residual stream is stored.
                g = lambda mod: mod.weight.data.to(device).clone().float()
                ld.update(norm3=g(b.pre_feedforward_layernorm),
                          norm4=g(b.post_feedforward_layernorm),
                          qn=g(a.q_norm), kn=g(a.k_norm), vn=self.no_gain,
                          scalar=b.layer_scalar.data.to(device).clone().float())
            elif self.gemma:
                ld["norm1"] = ld["norm1"] + 1.0
                ld["norm2"] = ld["norm2"] + 1.0
                ld.update(norm3=g1(b.pre_feedforward_layernorm),
                          norm4=g1(b.post_feedforward_layernorm),
                          qn=g1(a.q_norm) * (self.qn_fold or 1.0), kn=g1(a.k_norm))
            self.layers.append(ld)
            base.layers[i] = torch.nn.Identity()   # release the checkpoint copy as we go
        self.final_norm = base.norm.weight.data.to(device).clone().float()
        if self.gemma and not self.gemma4:
            self.final_norm = self.final_norm + 1.0
        self.tied = bool(getattr(c, "tie_word_embeddings", True))
        head_w = self.emb if self.tied else hf.lm_head.weight.data
        self.head = Linear(head_w, None, "lm_head", self.spec_for("lm_head"), ss.get("lm_head"),
                           device)
        del hf

        self.eps = self.store(torch.tensor(float(c.rms_norm_eps), device=device))
        self.inv_d = self.store(torch.tensor(1.0 / self.d, device=device))
        # Gemma's QK-norm reduces over the head dim, not the hidden size
        self.inv_hd = self.store(torch.tensor(1.0 / self.hd, device=device))
        self.layer_types = list(c.layer_types) if self.gemma else None
        self.sliding_window = int(c.sliding_window) if self.gemma else None
        self.softcap = ((float(getattr(c, "final_logit_softcapping", None) or 0.0) or None)
                        if self.gemma else None)
        if self.gemma:
            # one table per layer type: the local and global layers use two thetas,
            # and on Gemma-4 two head dims and two rope types as well
            self.inv_freqs = {t: self.inv_freq_of(
                dict(c.rope_parameters[t]),
                self.geoms[t]["hd"] if self.gemma4 else None).to(device)
                for t in set(self.layer_types)}
            self.inv_freq = None
        else:
            self.inv_freq = self.inv_freq_of(dict(c.rope_parameters)).to(device)
        if torch.device(device).type == "cuda":
            # the fp32 copies made while quantizing are freed; give the allocator's
            # reserve back so several models can share the GPU (B8's task lanes)
            torch.cuda.empty_cache()

    def inv_freq_of(self, rp, hd=None):
        """RoPE inverse frequencies from one `rope_parameters` entry, in fp64.

        `hd` overrides the model's head dim for Gemma-4, whose local and global
        layers rotate different widths; the default keeps every existing call
        bit-identical.
        """
        hd = self.hd if hd is None else hd
        theta = float(rp["rope_theta"])
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        rope_type = rp.get("rope_type", "default")
        if rope_type == "llama3":
            inv = llama3_inv_freq(inv, rp)
        elif rope_type == "proportional":
            inv = proportional_inv_freq(rp, hd, theta)
        elif rope_type != "default":
            raise NotImplementedError(f"rope_type {rope_type}")
        return inv

    def geometry_of(self, lc):
        """One layer type's head geometry, from its resolved per-layer config."""
        hd, nkv = int(lc.head_dim), int(lc.num_key_value_heads)
        return dict(hd=hd, nkv=nkv, rep=self.nh // nkv,
                    inv_hd=self.store(torch.tensor(1.0 / hd, device=self.device)))

    def use_geometry(self, layer_type):
        """Make one layer type's head geometry current.

        Everything downstream of the projections reads `self.hd`, `self.nkv`,
        `self.rep` and `self.inv_hd`, and on every family before Gemma-4 those are
        constants set once in `__init__`.  Gemma-4 is the first checkpoint where
        `q_proj`'s own output width differs between layer types, so the geometry is
        switched here rather than threaded through `rotate`, `attention` and
        `attention_int` -- which leaves the arithmetic the frozen specs were
        measured with untouched.
        """
        g = self.geoms[layer_type]
        self.hd, self.nkv, self.rep = g["hd"], g["nkv"], g["rep"]
        self.inv_hd = g["inv_hd"]

    # -- primitives -------------------------------------------------------
    def spec_for(self, kind):
        if kind in self.spec.fp32_matmuls:
            return replace(self.spec, linear="fp32")
        if kind in self.spec.pow2_matmuls:
            return replace(self.spec, pow2_scales="linear")
        return self.spec

    def store(self, x):
        return store(x, self.spec.nonlinear == "bf16")

    def dequant(self, acc, s_x, s_w):
        """`acc * s_x * s_w -> bf16`, per `Spec.dequant`.

        "fp32" is B1's original contract: form the product in fp32 and let the
        caller's `store()` round it once.  "bf16-chain" is what a gadget can prove
        when the scales are general bf16 values -- `bf16(bf16(bf16(acc) s_x) s_w)`,
        three correctly-rounded steps, two of them mantissa-product tables (B6
        measures that at 26 cells per accumulator against 15 for power-of-two
        scales).  When the scales *are* powers of two the two agree exactly, which
        is why `pow2-dequant` needs no separate numeric probe.
        """
        if self.spec.dequant == "bf16-chain":
            rnd = bf16_away if self.spec.dequant_round == "away" else self.store
            return rnd(rnd(rnd(acc) * s_x) * s_w)
        return acc * s_x * s_w

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

    def rmsnorm(self, x, w, key, inv_d=None):
        # `inv_d` overrides 1/hidden_size for Gemma's QK-norm, the same reduction
        # over the head dim; the default keeps every existing call bit-identical.
        ms = self.store(reduce_sum(self.store(x * x), -1, True, self.red("norm"))
                        * (self.inv_d if inv_d is None else inv_d))
        r = self.unary(f"{key}.rsqrt", self.store(ms + self.eps), torch.rsqrt)
        return self.store(self.store(x * r) * w)

    def rope(self, pos, layer_type=None):
        inv = self.inv_freq if layer_type is None else self.inv_freqs[layer_type]
        ang = pos.double().unsqueeze(-1) * inv.unsqueeze(0)
        ang = torch.cat([ang, ang], dim=-1)
        return self.store(ang.cos().float()), self.store(ang.sin().float())

    def rotate(self, t, cos, sin):
        h = self.hd // 2
        r = torch.cat([-t[..., h:], t[..., :h]], dim=-1)
        return self.store(self.store(t * cos) + self.store(r * sin))

    def attention_int(self, q, k, v, mask, li):
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
        sc = sc.masked_fill(mask.unsqueeze(0), -float(bound + 1))     # causal (and packing)
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
        if self.spec.attn_out_int8:
            # One rounding, straight to int8.  `o_proj` requantizes its input
            # anyway, so the softmax division, the value scale and that requantize
            # fold into a single `round(O s_v / (Z s_o))` -- which deletes the
            # reciprocal lookup, the bf16 waypoint and the requantize at every one
            # of 254 k output elements per token at 27B (B5 review; B6 prices the
            # gadget at 9 cells against 15 + 4 + 15).  The division is exact here
            # because O, Z are integers and the rounding happens once, at the end.
            real = o.double() * s_v.double() / z.double()
            flat = real.transpose(0, 1).reshape(ss, self.nh * self.hd)
            # stream Y: `attn_out_bits` widens the fold's target -- the same
            # single rounding, to a wider integer `o_proj` consumes as-is
            qmax_o = (1 << (self.spec.attn_out_bits - 1)) - 1
            s_o = qscale(flat.abs().amax(-1, keepdim=True).float(), -1, qmax_o,
                         self.spec.pow2_scales in ("all", "linear")
                         or self.spec.attn_out_pow2)
            q = round_hafz(flat / s_o.double()).clamp_(-qmax_o, qmax_o).float()
            if self.clamped is not None:
                self.tally("attnout", flat.float() / s_o, qmax_o)
            if self.rec is not None:
                self.rec.add(f"L{li}.attnq",
                             q.to(torch.int8 if qmax_o <= INT8_MAX else torch.int16))
                self.rec.add(f"L{li}.s_o", s_o.to(BF16))
                self.rec.add(f"L{li}.zz", z.transpose(0, 1).reshape(ss, self.nh)
                             .to(torch.int32))
            return None, (q, s_o)
        rz = self.unary(f"L{li}.recip", self.store(z), torch.reciprocal)
        out = self.store(o * s_v * rz)
        return out.transpose(0, 1).reshape(ss, self.nh * self.hd), None

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

    def attention(self, q, k, v, mask, li):
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
            return self.attention_int(q, k, v, mask, li)
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
        sc = sc.masked_fill(mask.unsqueeze(0), float("-inf"))      # causal (and packing)
        if self.rec is not None:
            self.rec.add(f"L{li}.scores", sc.to(BF16))
        e = self.unary(f"L{li}.exp", self.store(sc - sc.amax(-1, keepdim=True)), torch.exp)
        den = self.store(reduce_sum(e, -1, True, self.red("softmax")))
        p = self.store(e * self.unary(f"L{li}.recip", den, torch.reciprocal))
        if self.rec is not None:
            self.rec.add(f"L{li}.probs", p.to(BF16))
        out = self.store(p @ v)
        return out.transpose(0, 1).reshape(s, self.nh * self.hd), None

    # -- forward ----------------------------------------------------------
    def gemma_layer(self, i, l, x, cache, rot, mask):
        """One Gemma-3 decoder layer.

        The same seven matmuls, the same attention and the same recorder as the
        shared path; what differs is around them.  Four norms instead of two --
        the two extra ones normalize the attention output and the MLP output
        *before* the residual add -- Q and K are RMSNormed per head before RoPE,
        and the activation is GELU-tanh.  `rot` and `mask` are this layer type's
        (local or global) RoPE table and mask, chosen by the caller.
        """
        cos, sin = rot
        if self.rec is not None:
            self.rec.add(f"L{i}.xin", x.to(BF16))
        h = self.rmsnorm(x, l["norm1"], f"L{i}.norm1")
        q = l["q"](h, self).view(-1, self.nh, self.hd)
        k = l["k"](h, self).view(-1, self.nkv, self.hd)
        v = l["v"](h, self).view(-1, self.nkv, self.hd)
        q = self.rmsnorm(q, l["qn"], f"L{i}.qnorm", self.inv_hd)
        k = self.rmsnorm(k, l["kn"], f"L{i}.knorm", self.inv_hd)
        q = self.rotate(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = self.rotate(k, cos.unsqueeze(1), sin.unsqueeze(1))
        kk, vv = cache.append(i, k, v)
        a_out, a_pre = self.attention(q, kk, vv, mask, i)
        o_out = self.rmsnorm(l["o"](a_out, self, pre=a_pre), l["norm2"], f"L{i}.norm2")
        x = self.store(x + o_out)
        if self.rec is not None:
            self.rec.add(f"L{i}.o_out", o_out.to(BF16))
            self.rec.add(f"L{i}.xmid", x.to(BF16))
        h = self.rmsnorm(x, l["norm3"], f"L{i}.norm3")
        g = self.unary(f"L{i}.gelu", l["gate"](h, self), self.act)
        gu = g * l["up"](h, self)
        d_out = l["down"](gu if "up" in self.spec.delay_matmuls else self.store(gu), self)
        d_out = self.rmsnorm(d_out, l["norm4"], f"L{i}.norm4")
        x = self.store(x + d_out)
        if self.rec is not None:
            self.rec.add(f"L{i}.d_out", d_out.to(BF16))
            self.rec.add(f"L{i}.x", x.to(BF16))     # committed residual stream
        return x

    def gemma4_layer(self, i, l, x, cache, rot, mask):
        """One Gemma-4 text decoder layer.

        Gemma-3's frame -- four norms, QK-norm before RoPE, GELU-tanh, a RoPE table
        and a mask per layer type -- with the four things Gemma-3 does not have:

          * the head geometry is this layer type's; the caller has already made it
            current, which is what lets `q`, `k` and `v` reshape by `self.hd`;
          * `attention_k_eq_v`: on a global layer `l["v"]` is `None` and V is the
            *unnormalized* `k_proj` output -- HF takes V off `key_states` before
            `k_norm` runs -- so K and V share one projection and one matmul;
          * V passes through a scale-free RMSNorm (gain exactly 1.0) and is never
            rotated, on either layer type;
          * the layer's whole output, both residual branches included, is scaled by
            the per-layer `layer_scalar`.
        """
        cos, sin = rot
        if self.rec is not None:
            self.rec.add(f"L{i}.xin", x.to(BF16))
        h = self.rmsnorm(x, l["norm1"], f"L{i}.norm1")
        q = l["q"](h, self).view(-1, self.nh, self.hd)
        kv = l["k"](h, self).view(-1, self.nkv, self.hd)
        v = kv if l["v"] is None else l["v"](h, self).view(-1, self.nkv, self.hd)
        q = self.rmsnorm(q, l["qn"], f"L{i}.qnorm", self.inv_hd)
        k = self.rmsnorm(kv, l["kn"], f"L{i}.knorm", self.inv_hd)
        v = self.rmsnorm(v, l["vn"], f"L{i}.vnorm", self.inv_hd)
        q = self.rotate(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = self.rotate(k, cos.unsqueeze(1), sin.unsqueeze(1))
        kk, vv = cache.append(i, k, v)
        a_out, a_pre = self.attention(q, kk, vv, mask, i)
        o_out = self.rmsnorm(l["o"](a_out, self, pre=a_pre), l["norm2"], f"L{i}.norm2")
        x = self.store(x + o_out)
        if self.rec is not None:
            self.rec.add(f"L{i}.o_out", o_out.to(BF16))
            self.rec.add(f"L{i}.xmid", x.to(BF16))
        h = self.rmsnorm(x, l["norm3"], f"L{i}.norm3")
        g = self.unary(f"L{i}.gelu", l["gate"](h, self), self.act)
        gu = g * l["up"](h, self)
        d_out = l["down"](gu if "up" in self.spec.delay_matmuls else self.store(gu), self)
        d_out = self.rmsnorm(d_out, l["norm4"], f"L{i}.norm4")
        x = self.store(self.store(x + d_out) * l["scalar"])
        if self.rec is not None:
            self.rec.add(f"L{i}.d_out", d_out.to(BF16))
            self.rec.add(f"L{i}.x", x.to(BF16))     # committed residual stream
        return x

    def forward(self, ids, cache=None, pos=None, seq=None):
        """Logits for `ids`, appended to `cache`.

        By default the tokens continue the cache's single sequence.  `pos` and
        `seq` (one per token) place each token in its own sequence instead --
        B8's packed decode, one step of B sequences in one call (see `Cache`).
        """
        cache = cache or Cache(self.L)
        ids = ids.to(self.emb.device)
        if pos is None:
            pos = torch.arange(cache.n, cache.n + len(ids), device=ids.device)
            seq = torch.zeros(len(ids), dtype=torch.int64, device=ids.device)
        pos, seq = pos.to(ids.device), seq.to(ids.device)
        all_pos = pos if cache.pos is None else torch.cat([cache.pos, pos])
        all_seq = seq if cache.seq is None else torch.cat([cache.seq, seq])
        # (S, C): True where a key is hidden from a query -- another sequence, or
        # a later position (causal, absolute positions)
        mask = (all_seq.unsqueeze(0) != seq.unsqueeze(1)) | (all_pos.unsqueeze(0) > pos.unsqueeze(1))
        if self.gemma:
            # one RoPE table and one mask per layer type: a local layer sees only
            # the last `sliding_window` positions (a key is visible iff its
            # position is > the query's minus the window), a global one sees all
            rot = {t: self.rope(pos, t) for t in set(self.layer_types)}
            masks = {t: (mask if t == "full_attention" else
                         mask | (all_pos.unsqueeze(0) <= pos.unsqueeze(1) - self.sliding_window))
                     for t in set(self.layer_types)}
            x = self.store(self.emb[ids].float() * self.embed_scale)
        else:
            cos, sin = self.rope(pos)
            x = self.emb[ids].float()
        for i, l in enumerate(self.layers):
            if self.gemma:
                t = self.layer_types[i]
                if self.gemma4:
                    self.use_geometry(t)
                    x = self.gemma4_layer(i, l, x, cache, rot[t], masks[t])
                else:
                    x = self.gemma_layer(i, l, x, cache, rot[t], masks[t])
                continue
            # B10 step 2: the glue's witness is the residual stream at the three
            # points the two adds join, plus each add's second operand.  Nothing
            # numeric moves -- these are recordings of values the forward pass
            # already forms (`rec` is None outside an extract).
            if self.rec is not None:
                self.rec.add(f"L{i}.xin", x.to(BF16))
            h = self.rmsnorm(x, l["norm1"], f"L{i}.norm1")
            q = l["q"](h, self).view(-1, self.nh, self.hd)
            k = l["k"](h, self).view(-1, self.nkv, self.hd)
            v = l["v"](h, self).view(-1, self.nkv, self.hd)
            q = self.rotate(q, cos.unsqueeze(1), sin.unsqueeze(1))
            k = self.rotate(k, cos.unsqueeze(1), sin.unsqueeze(1))
            kk, vv = cache.append(i, k, v)
            a_out, a_pre = self.attention(q, kk, vv, mask, i)
            o_out = l["o"](a_out, self, pre=a_pre)
            x = self.store(x + o_out)
            if self.rec is not None:
                self.rec.add(f"L{i}.o_out", o_out.to(BF16))
                self.rec.add(f"L{i}.xmid", x.to(BF16))
            h = self.rmsnorm(x, l["norm2"], f"L{i}.norm2")
            g = self.unary(f"L{i}.silu", l["gate"](h, self), torch.nn.functional.silu)
            # stream Y: with up's requant delayed, the SwiGLU product also stays
            # exact -- down's per-token quantize is the chain's single rounding
            gu = g * l["up"](h, self)
            d_out = l["down"](gu if "up" in self.spec.delay_matmuls
                              else self.store(gu), self)
            x = self.store(x + d_out)
            if self.rec is not None:
                self.rec.add(f"L{i}.d_out", d_out.to(BF16))
                self.rec.add(f"L{i}.x", x.to(BF16))     # committed residual stream
        cache.append_meta(pos, seq)
        h = self.rmsnorm(x, self.final_norm, "final_norm")
        logits = self.head(h, self, store_out=not self.spec.fp32_logits)
        if self.softcap is not None:
            # Gemma's tanh softcap on the logits (null on Gemma-3-270M, 30.0 on
            # Gemma-4-31B); two more correctly-rounded ops, one table among them
            logits = self.store(self.unary("logit_cap", self.store(logits / self.softcap),
                                           torch.tanh) * self.softcap)
        return logits, cache
