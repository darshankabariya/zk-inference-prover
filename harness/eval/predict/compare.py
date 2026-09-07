"""E4, section 7 steps 6-7: the measured layer chunks against the registered prediction.

Reads the prover's per-layer JSONs (one `compose` run per layer, reference backend)
and the committed registration, writes the per-chunk record the run-book asks
for and the comparison, in the run-book's order:

  (a) the key tiles integers, exact -- reported whatever rho does;
  (b) rho = measured / predicted on layers-only fresh cells per decoded token,
      with section 3's protocol widening and no other adjustment;
  (c) the tier, and the consequence the bracket table fixed in advance.

The predicted figures are READ OUT OF `out/registration.json` (step 0c); nothing
here recomputes them.

    uv run python e4/compare.py --runs e4/out/h200 --shape qwen2p5_14b --ctx 4096
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "out", "registration.json")
SHAPES = {"qwen2p5_14b": "Qwen2.5-14B-Instruct",
          "qwen2p5_3b": "Qwen2.5-3B-Instruct",
          "qwen2p5_32b": "Qwen2.5-32B-Instruct"}


def load_run(path):
    r = json.load(open(path))
    p = r["result"]["point"]
    fc = p["family_cells"]
    ibf = fc["interaction_by_family"]
    pa = p["params"]
    k = int(re.search(r"L(\d+)\.json$", path).group(1))
    return dict(
        layer=k, file=os.path.basename(path),
        accepts=p["accepts"], prove_error=p.get("prove_error"),
        source=r["result"].get("source"),
        cells=dict(fresh=p["cells"]["fresh"], base=p["cells"]["base"],
                   interaction=p["cells"]["interaction"],
                   preprocessed=p["cells"]["preprocessed"]),
        columns=p["columns"],
        family_cells=dict(key_tiles=fc["key_tiles"], dequant=fc["dequant"],
                          residual_stream=fc["residual_stream"],
                          attnout=fc["attnout"], interaction=fc["interaction"]),
        interaction_by_family={f: ibf[f]["cells"] for f in ibf},
        params=dict(log_table=pa["log_table"], ao_rem_bits=pa["ao_rem_bits"],
                    ao_q_bits=pa["ao_q_bits"], key_tiles=pa["key_tiles"], nh=pa["nh"], hd=pa["hd"],
                    s=pa["s"], c=pa["c"], diagonal_block=pa.get("diagonal_block", False), layer_boundary=pa.get("layer_boundary", False),
                    grid_target=pa["grid_target"], chan_target=pa["chan_target"], rw=pa["rw"],
                    k_bits=pa["k_bits"], q_bits=pa["q_bits"], nmax=pa["nmax"],
                    kmax=pa["kmax"], round=pa["round"], variant=pa["variant"],
                    z_hi_bits=pa["z_hi_bits"], z_lo_bits=pa["z_lo_bits"],
                    residual_stream=pa.get("residual_stream"), mats=pa["mats"]),
        off_table=p.get("off_table"), ref_mismatch=p.get("ref_mismatch"),
        residual_stream_ties_ok=p.get("residual_stream_ties_ok"),
        peak_rss_bytes=p.get("peak_rss_bytes"),
        proof_sha256=p["proof_sha256"], proof_bytes=p["proof_bytes"],
        proof_ser_bytes=p["proof_ser_bytes"],
        statement_bytes=p.get("statement_bytes"),
        secs=dict(prove=p.get("prove_secs"), trace=p.get("trace_secs"),
                  interaction=p.get("interaction_secs"), verify=p.get("verify_secs"),
                  commit=(p.get("phases") or {}).get("commit_secs"),
                  stwo_wall=(p.get("phases") or {}).get("wall_secs"), fri=(p.get("phases") or {}).get("fri_secs")),
        env=r.get("env"),
    )


def tier_of(rho, br):
    if br["prize"][0] <= rho <= br["prize"][1]:
        return "prize"
    if br["pass_"][0] <= rho <= br["pass_"][1]:
        return "pass"
    if br["marginal"][0] <= rho <= br["marginal"][1]:
        return "marginal"
    return "fail"


CONSEQUENCE = {
    "prize": "the model predicts a never-fitted scale to 3%. Licence claimed unqualified; "
             "section 6.3 keeps [projected]",
    "pass": "licence claimed with the measured residual quoted beside it",
    "marginal": "licence claimed only with an explicit +/-25% key tile printed on every "
                "frontier row",
    "fail": "the cost model is refuted as an extrapolator. Section 6.3 regraded "
            "[projected] -> [illustrative]; the section 7 'you didn't build it' mitigation "
            "falls back to publishing the model runnable; E4 written up as a negative "
            "result. No re-fit may be presented as if it were this prediction.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of the prover's L<k>.json files")
    ap.add_argument("--shape", default="qwen2p5_14b")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--card", default="H200")
    ap.add_argument("--backend", default="ref (SimdBackend)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()

    reg = json.load(open(REG))
    shape = reg["shapes"][SHAPES[a.shape]]
    L = shape["cfg"][0]
    pc = shape["contexts"][str(a.ctx)]
    br = reg["bracket"]
    pred_cpt = pc["cells_per_token_layers"]
    pred_key_tiles_base = pc["key_tiles_family_base_per_chunk"]
    pred_key_tiles_inter = pc["key_tiles_family_interaction_per_chunk"]
    pred_chunk = pc["per_layer_chunk_fresh"]

    files = sorted(glob.glob(os.path.join(a.runs, "L*.json")),
                   key=lambda f: int(re.search(r"L(\d+)\.json$", f).group(1)))
    runs = [load_run(f) for f in files]
    if not runs:
        sys.exit(f"no L<k>.json under {a.runs}")

    # ---- step 5's two traps, per chunk ------------------------------------
    problems = []
    for r in runs:
        k = r["layer"]
        if not r["accepts"]:
            problems.append(f"L{k}: accepts != true")
        if not r["cells"]["fresh"] > 0:
            problems.append(f"L{k}: cells.fresh not > 0")
        if not r["family_cells"]["key_tiles"] > 0:
            problems.append(f"L{k}: family_cells empty")
        want = dict(q=(5120, 5120), k=(5120, 1024), v=(5120, 1024), o=(5120, 5120),
                    gate=(5120, 13824), up=(5120, 13824), down=(13824, 5120))
        if a.shape == "qwen2p5_14b":
            for m in r["params"]["mats"]:
                if (m["n_in"], m["n_out"]) != want[m["name"]] or m["variant"] != "chain" \
                        or m["tokens"] != 512:
                    problems.append(f"L{k}: mat {m}")
        p = r["params"]
        cfg_ok = (p["s"] == 512 and p["c"] == a.ctx and p["hd"] == 128 and p["diagonal_block"]
                  and p["layer_boundary"] and p["grid_target"] == 64 and p["chan_target"] == 128
                  and p["rw"] == 16 and p["k_bits"] == 11 and p["q_bits"] == 12
                  and p["round"] == "even" and p["nmax"] == 511)
        if not cfg_ok:
            problems.append(f"L{k}: configuration is not the registered one: {p}")
        if r["off_table"]:
            problems.append(f"L{k}: off_table {r['off_table']}")
        if r["ref_mismatch"]:
            problems.append(f"L{k}: ref_mismatch {r['ref_mismatch']}")
        if r["residual_stream_ties_ok"] is False:
            problems.append(f"L{k}: residual_stream_ties_ok false")

    # ---- (a) the key tiles integers, exact ------------------------------------
    key_tiles_rows = []
    for r in runs:
        b, i = r["family_cells"]["key_tiles"], r["interaction_by_family"]["key_tiles"]
        key_tiles_rows.append(dict(layer=r["layer"], key_tiles_base=b, key_tiles_interaction=i,
                               base_delta=b - pred_key_tiles_base,
                               interaction_delta=i - pred_key_tiles_inter,
                               exact=(b == pred_key_tiles_base and i == pred_key_tiles_inter)))
    key_tiles_exact = all(x["exact"] for x in key_tiles_rows)

    # ---- (b) rho, with the protocol widening ------------------------------
    n = len(runs)
    layers = [r["layer"] for r in runs]
    if n == L:
        protocol, widen_lo, widen_hi = "A", 0.0, 0.0
    elif n >= 8:
        gaps = sorted(set(b - a_ for a_, b in zip(layers, layers[1:])))
        protocol = "B" if len(gaps) == 1 else "B (not evenly spaced!)"
        widen_lo = widen_hi = br["protocol_widening"]["B_even_subset"]["widen_pct"]
    else:
        protocol = "C"
        widen_lo = br["protocol_widening"]["C_single_layer"]["widen_pct_lower"]
        widen_hi = br["protocol_widening"]["C_single_layer"]["widen_pct_upper"]
    fresh_sum = sum(r["cells"]["fresh"] for r in runs)
    fresh_mean = fresh_sum / n
    measured_cpt = fresh_mean * L / 512          # layers only, S = 512 decoded tokens
    rho = measured_cpt / pred_cpt
    widened = {t: [lo * (1 - widen_lo / 100), hi * (1 + widen_hi / 100)]
               for t, (lo, hi) in (("prize", br["prize"]), ("pass_", br["pass_"]),
                                   ("marginal", br["marginal"]))}
    tier = tier_of(rho, widened)

    # ---- the per-chunk record (step 6) ------------------------------------
    total_proof = sum(r["proof_bytes"] for r in runs)
    total_ser = sum(r["proof_ser_bytes"] for r in runs)
    rss = [r["peak_rss_bytes"] for r in runs if r["peak_rss_bytes"]]
    table_cells = [r["interaction_by_family"]["table"] for r in runs]
    logt = sorted(set(r["params"]["log_table"] for r in runs))
    # the 8B identity's analogue: fresh - key_tiles_fresh - 5 * 2^log_table, per layer
    rest = [r["cells"]["fresh"] - r["family_cells"]["key_tiles"]
            - r["interaction_by_family"]["key_tiles"] - 5 * (1 << r["params"]["log_table"])
            for r in runs]
    measured = dict(
        schema="e4-measured/1", shape=SHAPES[a.shape], hf=shape["hf"], cfg=shape["cfg"],
        ctx=a.ctx, s=512, spec="b10-final", card=a.card, backend=a.backend,
        boundary="cell count (card-independent); bytes; peak host RSS",
        layers_proved=layers, n_layers_proved=n, n_layers_model=L,
        chunks=runs,
        summary=dict(
            fresh_sum=fresh_sum, fresh_mean_per_chunk=fresh_mean,
            fresh_min=min(r["cells"]["fresh"] for r in runs),
            fresh_max=max(r["cells"]["fresh"] for r in runs),
            cells_per_token_layers_measured=measured_cpt,
            log_table_values=logt,
            log_table_histogram={str(t): sum(1 for r in runs if r["params"]["log_table"] == t)
                                 for t in logt},
            table_cells_min=min(table_cells), table_cells_max=max(table_cells),
            ao_rem_bits_values=sorted(set(r["params"]["ao_rem_bits"] for r in runs)),
            kmax_values=sorted(set(r["params"]["kmax"] for r in runs)),
            rest_after_key_tiles_and_table=dict(values=sorted(set(rest)), constant=len(set(rest)) == 1),
            # which families move across layers (at 8B every family but the table is
            # identical to the cell on all 32 chunks)
            family_spread={f: dict(min=min(r["family_cells"][f] for r in runs),
                                   max=max(r["family_cells"][f] for r in runs))
                           for f in ("key_tiles", "dequant", "residual_stream", "attnout", "interaction")},
            interaction_spread={f: dict(min=min(r["interaction_by_family"][f] for r in runs),
                                        max=max(r["interaction_by_family"][f] for r in runs))
                                for f in runs[0]["interaction_by_family"]},
            base_spread=dict(min=min(r["cells"]["base"] for r in runs),
                             max=max(r["cells"]["base"] for r in runs)),
            preprocessed_spread=dict(min=min(r["cells"]["preprocessed"] for r in runs),
                                     max=max(r["cells"]["preprocessed"] for r in runs)),
            proof_bytes_sum=total_proof, proof_ser_bytes_sum=total_ser,
            proof_bytes_per_layer_chunk_mean=total_proof / n,
            proof_ser_bytes_per_layer_chunk_mean=total_ser / n,
            proof_bytes_per_token_layers=total_proof / n * L / 512,
            proof_ser_bytes_per_token_layers=total_ser / n * L / 512,
            peak_rss_bytes_max=max(rss) if rss else None,
            peak_rss_bytes_min=min(rss) if rss else None,
            prove_secs_sum=sum(r["secs"]["prove"] or 0 for r in runs),
            trace_secs_sum=sum(r["secs"]["trace"] or 0 for r in runs),
        ),
        problems=problems,
    )
    comparison = dict(
        schema="e4-comparison/1", shape=SHAPES[a.shape], ctx=a.ctx, s=512, spec="b10-final",
        card=a.card, backend=a.backend,
        registration=dict(file=os.path.relpath(REG, HERE), prediction_sha256=reg["prediction_sha256"],
                          prediction_block_sha256=reg["prediction_block_sha256"]),
        predicted=dict(cells_per_token_layers=pred_cpt, per_layer_chunk_fresh=pred_chunk,
                       key_tiles_family_base_per_chunk=pred_key_tiles_base,
                       key_tiles_family_interaction_per_chunk=pred_key_tiles_inter,
                       cells_per_token_layers_key_tiles=pc["cells_per_token_layers_key_tiles"],
                       cells_per_token_layers_rest=pc["cells_per_token_layers_rest"]),
        a_key_tiles_exact=dict(exact_on_every_chunk=key_tiles_exact, rows=key_tiles_rows),
        b_rho=dict(protocol=protocol, layers_proved=n, widen_pct_lower=widen_lo,
                   widen_pct_upper=widen_hi, measured_cells_per_token_layers=measured_cpt,
                   predicted_cells_per_token_layers=pred_cpt, rho=rho,
                   residual_pct=(rho - 1) * 100,
                   chunk_ratio_min=min(r["cells"]["fresh"] for r in runs) / pred_chunk,
                   chunk_ratio_max=max(r["cells"]["fresh"] for r in runs) / pred_chunk,
                   brackets_widened=widened),
        c_tier=dict(tier=tier, consequence=CONSEQUENCE[tier]),
        void=bool(problems), problems=problems,
    )
    os.makedirs(a.out, exist_ok=True)
    mp = os.path.join(a.out, f"measured_{a.shape}_c{a.ctx}.json")
    cp = os.path.join(a.out, f"comparison_{a.shape}_c{a.ctx}.json")
    json.dump(measured, open(mp, "w"), indent=1, sort_keys=True)
    json.dump(comparison, open(cp, "w"), indent=1, sort_keys=True)

    # ---- the table ----------------------------------------------------------
    print(f"E4 -- {SHAPES[a.shape]} at C={a.ctx}, S=512, spec b10-final, {a.card}, {a.backend}")
    print(f"registration: prediction_sha256 {reg['prediction_sha256'][:12]}...  "
          f"predicted cells/token (layers) {pred_cpt:,.1f}, per chunk {pred_chunk:,.1f}")
    print()
    print(f"{'L':>3} {'fresh':>14} {'key_tiles':>12} {'i.key tiles':>12} {'table':>12} {'logT':>4} "
          f"{'aorem':>5} {'kmax':>4} {'proof_B':>10} {'ser_B':>11} {'rss_GiB':>7} {'prove_s':>7} digest")
    for r in runs:
        print(f"{r['layer']:>3} {r['cells']['fresh']:>14,} {r['family_cells']['key_tiles']:>12,} "
              f"{r['interaction_by_family']['key_tiles']:>12,} {r['interaction_by_family']['table']:>12,} "
              f"{r['params']['log_table']:>4} {r['params']['ao_rem_bits']:>5} {r['params']['kmax']:>4} "
              f"{r['proof_bytes']:>10,} {r['proof_ser_bytes']:>11,} "
              f"{(r['peak_rss_bytes'] or 0) / 2**30:>7.1f} {(r['secs']['prove'] or 0):>7.1f} "
              f"{r['proof_sha256'][:16]}")
    print()
    print(f"(a) key tiles exact on every chunk: {key_tiles_exact}  "
          f"(registered {pred_key_tiles_base:,} / {pred_key_tiles_inter:,})")
    if not key_tiles_exact:
        for x in key_tiles_rows:
            if not x["exact"]:
                print(f"    L{x['layer']}: base {x['key_tiles_base']:,} ({x['base_delta']:+,}), "
                      f"interaction {x['key_tiles_interaction']:,} ({x['interaction_delta']:+,})")
    print(f"(b) protocol {protocol}: {n}/{L} layers; measured {measured_cpt:,.1f} cells/token "
          f"(layers), predicted {pred_cpt:,.1f}; rho = {rho:.5f} ({(rho - 1) * 100:+.3f}%)")
    print(f"    per-chunk ratio range {comparison['b_rho']['chunk_ratio_min']:.4f} .. "
          f"{comparison['b_rho']['chunk_ratio_max']:.4f}; log_table {logt}; "
          f"rest-after-key-tiles-and-table constant: {len(set(rest)) == 1} {sorted(set(rest))[:4]}")
    print(f"(c) tier: {tier.upper()} -- {CONSEQUENCE[tier]}")
    sp = measured["summary"]
    moving = {f: v for f, v in {**{"family." + k: v for k, v in sp["family_spread"].items()},
                                **{"interaction." + k: v for k, v in sp["interaction_spread"].items()}}.items()
              if v["min"] != v["max"]}
    print(f"families that move across layers: "
          f"{ {f: v['max'] - v['min'] for f, v in moving.items()} or 'none'}")
    print(f"proof bytes: {total_proof:,} over {n} chunks ({total_proof / n:,.0f}/chunk, "
          f"{total_proof / n * L / 512:,.0f} B/token layers-only); ser {total_ser:,} "
          f"({total_ser / n:,.0f}/chunk, {total_ser / n * L / 512:,.0f} B/token)")
    if rss:
        print(f"peak RSS: {max(rss) / 2**30:.1f} GiB max, {min(rss) / 2**30:.1f} GiB min")
    if problems:
        print("VOID -- the run does not reproduce the registered configuration:")
        for p in problems:
            print("   ", p)
    print(f"-> {mp}\n-> {cp}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
