"""E1-E3: arithmetic feasibility of the 'int8 matmul in M31 + FP nonlinears via lookups' theory.

Model proxy: Gemma-3-27B-like dense transformer (public config), scaled notes for ~31B.
"""
import numpy as np

P = (1 << 31) - 1  # Mersenne-31

# ---------------- E1: int8 matmul headroom in M31 ----------------
print("=" * 72)
print("E1: int8 matmul accumulation headroom in M31 (p = 2^31 - 1)")
print("=" * 72)
# symmetric dynamic quantization: activations/weights in [-127, 127]
half = (P - 1) // 2  # max magnitude representable for signed values mod p
kmax = half // (127 * 127)
print(f"max |dot| for K-dim int8 dot = 127^2*K;  fits signed-in-M31 iff K <= {kmax}")
dims = {
    "attn QK^T (K=head_dim=128)": 128,
    "attn out-proj (K=n_heads*hd=4096)": 4096,
    "hidden matmuls (K=d_model=5376)": 5376,
    "FFN down-proj (K=d_ff=21504)": 21504,
    "attn AV (K=ctx=16384)": 16384,
    "attn AV (K=ctx=65536)": 65536,
    "hypothetical K=70000": 70000,
}
for name, K in dims.items():
    worst = 127 * 127 * K
    print(f"  {name:38s} worst |acc| = {worst:>14,}  "
          f"{'OK  (fits one M31 elem)' if worst <= half else 'OVERFLOW'}")

# empirical: exact equality of M31 field arithmetic vs int64 reference at real dims
rng = np.random.default_rng(0)
for k in (5376, 21504, 16384):
    a = rng.integers(-127, 128, size=(4, k))
    b = rng.integers(-127, 128, size=(k, 4))
    ref = a @ b
    for i in range(4):
        for j in range(4):
            acc = 0
            for t in range(k):  # true field arithmetic: reduce mod p at every step
                acc = (acc + (int(a[i, t]) % P) * (int(b[t, j]) % P)) % P
            lifted = acc - P if acc > half else acc
            assert lifted == ref[i, j]
    print(f"  empirical check K={k:>6}: M31 field arithmetic == exact int64 dot  \u2713")

# ---------------- E2: op census per decoded token ----------------
print()
print("=" * 72)
print("E2: op census per decoded token (Gemma-3-27B-like proxy config)")
print("=" * 72)
L, d, nh, nkv, hd, dff, vocab = 62, 5376, 32, 16, 128, 21504, 262144
params = vocab*d + L*(d*nh*hd + 2*d*nkv*hd + nh*hd*d + 3*d*dff)
print(f"config: L={L} d={d} heads={nh}/{nkv}kv hd={hd} dff={dff} vocab={vocab//1024}K "
      f"-> {params/1e9:.1f}B params")

for ctx in (4096, 16384):
    weight_macs = params  # one MAC per weight per token (batch-1 decode)
    attn_macs = L * (nh*hd*ctx + nh*hd*ctx)          # QK^T + AV per token (avg ~ctx/2; use full ctx = upper bound)
    macs = weight_macs + attn_macs
    # nonlinear elementwise FP ops per token
    softmax_exp = L * nh * ctx                        # exp per attention row
    rmsnorm     = L * 2 * (2*d + 1)                   # square+mul + rsqrt
    geglu       = L * dff                             # gelu evals
    rotary      = L * (nh + nkv) * hd                 # sin/cos muls
    logits_sm   = vocab                               # final softmax/sampling
    nonlin = softmax_exp + rmsnorm + geglu + rotary + logits_sm
    print(f"\nctx={ctx}:")
    print(f"  matmul MACs/token:        {macs/1e9:8.1f} G   (weights {weight_macs/1e9:.1f}G + attn {attn_macs/1e9:.1f}G)")
    print(f"  nonlinear FP ops/token:   {nonlin/1e6:8.1f} M   "
          f"(softmax {softmax_exp/1e6:.1f}M, geglu {geglu/1e6:.2f}M, norms {rmsnorm/1e3:.0f}K)")
    print(f"  ratio nonlin/MACs:        {nonlin/macs*100:8.4f} %")
    for gadget in (8, 32, 128):
        print(f"    if FP-op gadget costs {gadget:3d} field ops -> overhead vs MACs: "
              f"{nonlin*gadget/macs*100:6.2f} %")

# lookup-table domain sizes for FP primitives
print("\nlookup-table domains (LogUp-style) for FP primitives:")
fmts = {"fp8 e4m3": 8, "bf16": 16, "fp16": 16, "fp32": 32}
for f, bits in fmts.items():
    print(f"  unary nonlinear on {f:9s}: table 2^{bits}  "
          f"({'trivial' if bits<=16 else 'needs decomposition'})")
print("  binary op on bf16 pair: 2^32 domain -> decompose: mantissa8 x mantissa8 mul = 2^16 table,")
print("  exponent align/add via small range tables; i.e. bf16 arithmetic = a few 2^16 lookups/op.")

# ---------------- E3: prover budget implied by 53 tok/s on one H100 ----------------
print()
print("=" * 72)
print("E3: prover budget at published throughput (single H100)")
print("=" * 72)
H100_IMAD = 132 * 64 * 1.98e9        # SMs x int32 lanes x clock ~ 1.67e13 int32 mul-add/s
H100_INT8_TC = 1.979e15              # int8 tensor-core OPS (dense, with sparsity 2x more)
for ctx, tps in ((16384, 53.0), (4096, 77.0)):
    attn_macs = L * 2 * nh * hd * ctx
    macs = (params + attn_macs) * tps
    print(f"\nctx={ctx}, {tps} tok/s -> {macs/1e12:.2f} T MACs proven per second")
    for c in (1, 2, 4, 8, 16):
        frac = macs * c * 3 / H100_IMAD   # ~3 IMAD per M31 mul-add (mul+reduce+add)
        print(f"  {c:2d} field-ops/MAC -> {frac*100:7.1f} % of H100 int32 throughput"
          + ("   <-- infeasible" if frac > 1 else ""))
    print(f"  (same work on int8 tensor cores, if field ops mapped there: "
          f"{macs*2*3/H100_INT8_TC*100:.2f} % at 2 ops/MAC)")

print("""
commitment-budget contrast (per token, ctx=16K):
  commit EVERY mult wire (zkVM/AIR style):  ~35 G values/token  = ~140 GB/token  -> impossible
  commit layer I/O only (GKR style):        ~L*6*d ~ 2.0 M values/token = ~8 MB/token
                                            at 53 tok/s -> ~0.4 GB/s to hash  -> trivial on GPU
  attention softmax lookup columns:         L*nh*ctx ~ 32.5 M values/token = 130 MB/token
                                            at 53 tok/s -> ~6.9 GB/s hash+FRI  -> the memory hog
""")
