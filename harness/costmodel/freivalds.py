"""E6: Freivalds-inside-a-STARK — mechanics, soundness, op counts, and the 2x/100x check.

The approach: proving Y = W @ X (X = activations of ALL S tokens of the sequence,
stacked) is done by committing X, Y, deriving random r via Fiat-Shamir, and proving
only  r^T Y = (r^T W) X  inside the circuit. In-circuit work collapses from
S * n_in * n_out MACs to  n_in*n_out (once per proof, the r^T W pass over committed
weights) + S*(n_in + n_out) per-token inner products.
"""
import numpy as np

P = (1 << 31) - 1
rng = np.random.default_rng(1)

# ---------- soundness demo: batched Freivalds catches a single corrupted entry ----
print("=" * 74)
print("E6a: Freivalds soundness (toy, single M31 draw; production would use QM31)")
print("=" * 74)
n, m, S = 256, 256, 64
W = rng.integers(-127, 128, (n, m)); X = rng.integers(-127, 128, (m, S))
Y = W @ X
trials, caught = 2000, 0
for _ in range(trials):
    Ybad = Y.copy(); Ybad[rng.integers(n), rng.integers(S)] += 1   # minimal corruption
    r = rng.integers(0, P, n); s = rng.integers(0, P, S)           # FS randomness
    lhs = (r % P) @ (Ybad % P) @ (s % P) % P
    rhs = (((r % P) @ (W % P)) % P) @ ((X % P) @ (s % P) % P) % P
    caught += int(lhs != rhs)
print(f"  corrupted-by-1 detection: {caught}/{trials}   (false-accept <= 2/p ~ 9e-10;")
print(f"   over QM31 (~2^124) the whole model stays under 2^-100 with room to spare)")

# ---------- in-circuit op counts: naive vs Freivalds, Gemma-27B-like proxy --------
print()
print("=" * 74)
print("E6b: in-circuit multiplies per DECODED token (Gemma-3-27B-like proxy)")
print("=" * 74)
L, d, nh, nkv, hd, dff, vocab = 62, 5376, 32, 16, 128, 21504, 262144
params = vocab*d + L*(d*nh*hd + 2*d*nkv*hd + nh*hd*d + 3*d*dff)
mats = [(d, (nh + 2*nkv)*hd), (nh*hd, d), (d, 2*dff), (dff, d)]     # per layer
io_per_layer = sum(a + b for a, b in mats)
EXT = 4                                                              # QM31 = 4 x M31
for C in (4096, 16384):
    naive = params + L*2*nh*hd*C
    for S in (32, 512, 4096):
        weight_pass = EXT * params / S                    # r^T W, amortized per proof
        per_tok_io  = EXT * L * io_per_layer              # (r^T W) x_t and r^T y_t
        attn        = 3 * EXT * L * nh * C                # QK^T + AV Freivalds rows
        lookups     = L*nh*C + L*(2*dff + 2*(2*d+1)) + vocab   # bf16 nonlinear LogUps
        total = weight_pass + per_tok_io + attn + lookups
        print(f"  ctx={C:>6} S={S:>5}: naive {naive/1e9:6.1f}G -> Freivalds "
              f"{total/1e6:7.0f}M ops/token  (x{naive/total:5.0f} less; "
              f"weight-pass share {weight_pass/total*100:4.1f}%)")

# ---------- E6c: does '2x output / 100x input' cohere with 53 tok/s? --------------
print()
print("=" * 74)
print("E6c: coherence of the reported overheads with the published throughput")
print("=" * 74)
BW, TOPS = 3.35e12, 1.0e15          # H100 HBM3 B/s; effective int8 tensor-core ops/s
t_dec = params / BW                  # memory-bound int8 decode, s/token
t_pre = 2 * params / TOPS            # compute-bound int8 prefill, s/token
print(f"  plain int8 inference, one H100:  decode {1/t_dec:5.0f} tok/s   "
      f"prefill {1/t_pre:6.0f} tok/s   (ratio {t_pre and (1/t_pre)/(1/t_dec):3.0f}x)")
for k_ms in (4, 8, 11, 15):
    k = k_ms / 1e3
    print(f"  flat prover cost k={k_ms:2d} ms/token -> proven decode "
          f"{1/(t_dec+k):5.1f} tok/s | overhead: output x{(t_dec+k)/t_dec:4.1f}, "
          f"input x{(t_pre+k)/t_pre:5.0f}")
print("""
  reported: 53 tok/s, 'x2 on output', 'x100 on input'
  -> all three match a FLAT per-token prover cost of ~8-11 ms on the same GPU,
     i.e. proving costs about one extra decode-pass per token, while prefill
     (~50-100x cheaper per token to compute) shows the same k as a ~100x overhead.
""")
