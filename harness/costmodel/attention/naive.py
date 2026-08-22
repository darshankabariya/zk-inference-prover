"""B9 step 2, probe 1: the naive baseline, closed on measured constants.

The naive prover commits one trace cell per MAC wire -- every multiply and every
add of every matmul and of the attention grid -- instead of proving the matmul by
a Freivalds identity over a random challenge. Nothing here is run: the point of
the probe is that the number is so far outside the machine that running it is not
a measurement, it is a category error. Every constant is measured somewhere else
and cited.

    uv run experiments/b9/naive.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"
B2 = HERE.parent / "b2" / "out"

# (L, d, nh, nkv, hd, dff, vocab)
SHAPES = {
    "Llama-3.1-8B": (32, 4096, 32, 8, 128, 14336, 128256),
    "Gemma-3-27B (the 27B proxy shape)": (62, 5376, 32, 16, 128, 21504, 262144),
}
CTXS = (4096, 16384)
TARGET_TOKS = 53.0

# --- measured constants, all cited --------------------------------------------
# B7b, the largest chunk (2.217 G cells, the 7B shape at C=16K; b7/README 26.5)
GPU_COMMIT_CELLS_S = 4.45e9
GPU_WALL_CELLS_S = 1.46e9
# B2, one CPU in the container (b2/out/commit.json, the peak over the wide traces)
CPU_COMMIT_CELLS_S = json.loads((B2 / "commit.json").read_text())["result"]["points"]
CPU_COMMIT_CELLS_S = max(p["base_cells_per_sec"]["total"]
                         for p in CPU_COMMIT_CELLS_S if p["n_cols"] >= 16)
# H100 SXM HBM3, the machine every GPU number above was measured on
HBM_BYTES_S = 3.35e12
# B7 section 3: bytes the prove path touches per base cell at blowup 2 -- write,
# interpolate, LDE, Merkle input, OODS, quotient, constraint reads
BYTES_PER_CELL = 60.0
M31_BYTES = 4.0
# B9 step 0: our own committed cells per token, the model's default layout
OURS_PER_TOKEN = {4096: 188.22e6, 16384: 300.97e6}      # 27B; printed by b2/cost_model.py


def n_params(cfg):
    L, d, nh, nkv, hd, dff, vocab = cfg
    return vocab * d + L * (d * nh * hd + 2 * d * nkv * hd + nh * hd * d + 3 * d * dff)


def naive_cells(cfg, ctx):
    """One committed cell per MAC wire: the linear MACs are one per weight per
    token, and the attention grid is Q.K^T and A.V, both nh*hd*ctx per token."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    return n_params(cfg) + L * 2 * nh * hd * ctx


def g(x, unit=""):
    for s, dv in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= dv:
            return f"{x / dv:.2f} {s}{unit}"
    return f"{x:.2f} {unit}"


rows = []
for name, cfg in SHAPES.items():
    for ctx in CTXS:
        nc = naive_cells(cfg, ctx)
        rows.append(dict(
            shape=name, params=n_params(cfg), ctx=ctx, naive_cells_per_token=nc,
            # the trace itself, one M31 per cell
            trace_gb_per_token=nc * M31_BYTES / 1e9,
            # and what the prove path actually moves for it
            touched_gb_per_token=nc * BYTES_PER_CELL / 1e9,
            cells_s_for_53=nc * TARGET_TOKS,
            bytes_s_for_53=nc * TARGET_TOKS * BYTES_PER_CELL,
            x_hbm=nc * TARGET_TOKS * BYTES_PER_CELL / HBM_BYTES_S,
            x_gpu_commit=nc * TARGET_TOKS / GPU_COMMIT_CELLS_S,
            x_gpu_wall=nc * TARGET_TOKS / GPU_WALL_CELLS_S,
            x_cpu_commit=nc * TARGET_TOKS / CPU_COMMIT_CELLS_S,
            tok_s_on_h100=GPU_WALL_CELLS_S / nc,
        ))

print("=" * 100)
print("B9 probe 1 -- the naive baseline: one committed cell per MAC wire")
print("=" * 100)
print(f"measured constants: H100 commit path {g(GPU_COMMIT_CELLS_S)}cells/s, whole wall "
      f"{g(GPU_WALL_CELLS_S)}cells/s (B7b, 2.217 G chunk);")
print(f"                    one CPU {g(CPU_COMMIT_CELLS_S)}cells/s (B2); HBM3 "
      f"{g(HBM_BYTES_S)}B/s; {BYTES_PER_CELL:.0f} B touched per base cell (B7 section 3)")
print()
hdr = (f"{'shape':<34} {'C':>6} {'naive cells/tok':>16} {'trace GB/tok':>13} "
       f"{'B/s for 53':>12} {'x HBM':>8} {'x commit':>9} {'tok/s':>9}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['shape']:<34} {r['ctx']:>6} {g(r['naive_cells_per_token']):>16} "
          f"{r['trace_gb_per_token']:>13.1f} {g(r['bytes_s_for_53']):>12} "
          f"{r['x_hbm']:>8,.0f} {r['x_gpu_commit']:>9,.0f} {r['tok_s_on_h100']:>9.4f}")
print()
print("against the scheme as built, at 27B:")
for ctx in CTXS:
    r = next(x for x in rows if x["ctx"] == ctx and x["shape"].startswith("Gemma"))
    ours = OURS_PER_TOKEN[ctx]
    print(f"  C={ctx:<6} naive {g(r['naive_cells_per_token'])}cells/token vs ours "
          f"{g(ours)}  ->  x{r['naive_cells_per_token'] / ours:,.0f} fewer cells, and "
          f"{r['trace_gb_per_token']:.1f} GB/token of trace vs "
          f"{ours * M31_BYTES / 1e9:.2f} GB")
print()
print("what the three earlier blocks each say about the same gap:")
print("  B2  the naive prover needs 7.5 TB/s of trace at 27B/16K -- 3-4 orders of")
print("      magnitude past the measured commit path; the Freivalds structure is not")
print("      an optimisation, it is what makes the problem exist (b2/README, verdicts)")
print("  B3  the weight pass is the term that collapses: the r^T W walk commits")
print("      0.090 fresh cells per weight against one per MAC, and in-circuit")
print("      multiplies drop 31x at S=32, 305x at S=512, 639-1365x at S=4096 --")
print("      the ratio grows with S because only the weight pass is fixed (b3 section 8)")
print("  B4  what is left is not free: reading an accumulator as an integer rather")
print("      than a field element costs 5 cells (26.12 with B6's rounding), 9.8% of")
print("      the per-token trace at 27B, for a statement a field-only argument gets")
print("      wrong by exactly p (b4 section 9)")
print()
v = next(x for x in rows if x["ctx"] == 16384 and x["shape"].startswith("Gemma"))
vours = OURS_PER_TOKEN[16384]
print("verdict, at 27B/C=16K:")
print(f"  the naive baseline is {g(v['naive_cells_per_token'])}cells/token -- "
      f"x{v['naive_cells_per_token'] / vours:,.0f} our {g(vours)} -- i.e. "
      f"{v['trace_gb_per_token']:.0f} GB of trace per token against "
      f"{vours * M31_BYTES / 1e9:.2f} GB.")
print(f"  53 tok/s of it is {g(v['cells_s_for_53'] * M31_BYTES)}B/s of trace *written* "
      f"(B2's 7.5 TB/s, reproduced)")
print(f"  and {g(v['bytes_s_for_53'])}B/s *touched* on the prove path -- "
      f"x{v['x_hbm']:,.0f} an H100's HBM, before a single hash.")
print(f"  On the measured commit path it is x{v['x_gpu_commit']:,.0f} short; on the whole "
      f"wall, {v['tok_s_on_h100']:.4f} tok/s.")
print("  Three to four orders of magnitude is not an engineering gap.  Every other")
print("  number in this block -- x3.6 on the commit path, x11 on the wall -- exists")
print("  only because the Freivalds identity took this one off the table first.")

OUT.mkdir(exist_ok=True)
(OUT / "naive.json").write_text(json.dumps(dict(
    constants=dict(gpu_commit_cells_s=GPU_COMMIT_CELLS_S, gpu_wall_cells_s=GPU_WALL_CELLS_S,
                   cpu_commit_cells_s=CPU_COMMIT_CELLS_S, hbm_bytes_s=HBM_BYTES_S,
                   bytes_per_cell=BYTES_PER_CELL, target_toks=TARGET_TOKS),
    ours_per_token_27b=OURS_PER_TOKEN, rows=rows), indent=1))
print()
print(f"wrote {OUT / 'naive.json'}")
