"""B1 numeric primitives: bf16 values, int8 quantization, exact integer GEMM.

Numeric contract of the attested computation (B1 README spells out the rationale):

  * every value flowing between ops is a bf16 value.  Tensors are carried in fp32
    for BLAS speed, but `store()` is applied at every op boundary so the *values*
    are always exactly bf16;
  * elementwise unary/binary ops are correctly-rounded bf16 -- a unary op is
    exactly a 2^16-entry LogUp table, a binary one a small-table FP gadget;
  * the RMSNorm mean and the softmax denominator accumulate in fp32 and round
    once -- the v1 soundness gap B4 allows.  With `reduce="bf16"` they instead use
    a balanced binary tree of correctly-rounded bf16 adds, which a v2 FP-add
    gadget could prove directly.  The two bf16 attention matmuls always
    accumulate in fp32;
  * weight matmuls are exact int8 x int8 -> int32 accumulations; the dequant
    acc * s_x * s_w (+ bias) is evaluated in fp32 and rounded once to bf16.

Quantization is symmetric everywhere: per-output-channel for weights, per-token
(per row) dynamic for activations, round-half-away-from-zero, clamp to +-127.
Scales are themselves bf16 values, rounded *up* so that the tensor's own maximum
never hits the clamp.
"""
import torch

BF16 = torch.bfloat16
INT8_MAX = 127
# fp32 represents integers exactly below 2^24; a K-term int8 dot product can
# reach 127^2*K, so wider reductions have to go through dgemm (exact below 2^53).
SGEMM_MAX_K = (1 << 24) // (INT8_MAX * INT8_MAX)


def store(x, bf16=True):
    """Round an op result to the value that gets committed."""
    return x.to(BF16).float() if bf16 else x


def bf16_ceil(x):
    """Smallest bf16 value >= x, for x > 0."""
    b = x.to(BF16)
    bump = (b.float() < x).to(torch.int16)
    return (b.contiguous().view(torch.int16) + bump).view(BF16).float()


def round_hafz(x):
    """Round half away from zero (torch.round is half-to-even)."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def tree_sum(x, dim=-1):
    """Balanced-tree sum with every partial sum rounded to bf16.

    A fixed, documented order: pad to a power of two with +0, then log2(n) rounds
    of correctly-rounded pairwise adds.  Cheaper and more accurate than a
    sequential sum, and each add is one FP-add gadget invocation.
    """
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    pad = 1 << (n - 1).bit_length()
    if pad != n:
        x = torch.nn.functional.pad(x, (0, pad - n))
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = store(x[..., :h] + x[..., h:])
    return x.movedim(-1, dim)


def reduce_sum(x, dim=-1, keepdim=False, mode="fp32"):
    if mode == "bf16":
        s = tree_sum(x, dim)
        return s if keepdim else s.squeeze(dim)
    return x.sum(dim=dim, keepdim=keepdim)


def qscale_per_token(x, pow2=False, eps=1e-30):
    """Dynamic symmetric activation scale, one per token (per row)."""
    return qscale(x, -1, INT8_MAX, pow2, eps)


def qscale_per_channel(w, pow2=False, eps=1e-30):
    """Symmetric weight scale, one per output channel.  w is (n_in, n_out)."""
    return qscale(w, 0, INT8_MAX, pow2, eps)


def quantize(x, s, qmax=INT8_MAX):
    """Integer values (held in fp32) for x under scale s."""
    return round_hafz(x / s).clamp_(-qmax, qmax)


def gemm_dtype(k):
    return torch.float32 if k <= SGEMM_MAX_K else torch.float64


def int_gemm(qx, qw):
    """Exact integer matmul of int8-valued operands via BLAS.

    Every partial sum is an integer bounded by 127^2*K, which the operand dtype
    (chosen by `gemm_dtype`) represents exactly, so this is bit-identical to an
    int64 matmul -- `selftest.py` checks that, and the M31 embedding, on real data.
    """
    return (qx.to(qw.dtype) @ qw).float()


def pow2_ceil(x):
    """Smallest power of two >= x, exactly (x > 0).

    A power-of-two scale turns every dequant multiply into an exponent add, which
    is what lets B5's exp-table input be read off a window of the score's limbs
    instead of proven through a mantissa product (plan, B5 constraints).
    """
    m, e = torch.frexp(x)                       # x = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)         # x already a power of two
    return torch.ldexp(torch.ones_like(x), e)


def qscale(x, dim=-1, qmax=INT8_MAX, pow2=False, eps=1e-30):
    """Symmetric max-scale over `dim`, as a bf16 (or power-of-two) value."""
    a = x.abs().amax(dim=dim, keepdim=True).clamp_min(eps) / qmax
    return pow2_ceil(a) if pow2 else bf16_ceil(a)


def int_gemm_exact(qa, qb, bound):
    """Exact integer matmul whose partial sums are bounded by `bound`.

    fp32 holds integers exactly below 2^24 and fp64 below 2^53, so the operand
    dtype is chosen from the bound rather than from the reduction width alone --
    B5's numerator matmul reaches 255 * 127 * C, not 127^2 * C.
    """
    dt = torch.float32 if bound < (1 << 24) else torch.float64
    return (qa.to(dt) @ qb.to(dt)).float()
