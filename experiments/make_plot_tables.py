#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
import pandas as pd


def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _stderr(series: pd.Series) -> float:
    s = series.dropna()
    n = len(s)
    if n <= 1:
        return float("nan")
    return float(s.std(ddof=1) / math.sqrt(n))


def main():
    ap = argparse.ArgumentParser(
        description="Build compact CSV tables used to plot curves (runtime, cost) from aggregated summary CSV."
    )
    ap.add_argument("input_csv", help="Aggregated CSV (e.g., experiments/summary_all_camera.csv)")
    ap.add_argument("--out", default="experiments/plot_table_used_for_figures.csv",
                    help="Output plot table CSV")
    ap.add_argument("--by", default="graph_size",
                    help="Group key (default graph_size). Use 'flow' for real flows.")
    ap.add_argument("--x", default="graph_size",
                    help="X axis column name to keep (default graph_size).")
    ap.add_argument("--include-std", action="store_true",
                    help="Also output std/stderr columns.")
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.out)

    df = pd.read_csv(in_path)

    # Normalisation colonnes attendues (selon tes fichiers)
    # runtime
    if "runtime_sec" in df.columns:
        df["runtime_sec"] = df["runtime_sec"].apply(_safe_float)

    # coûts : selon tes versions, ça peut être raw_cost / normalized_cost / weighted_cost / relative_cost_vs_naive
    cost_cols = [c for c in [
        "weighted_cost", "raw_cost", "normalized_cost", "relative_cost_vs_naive", "relative_cost"
    ] if c in df.columns]

    for c in cost_cols:
        df[c] = df[c].apply(_safe_float)

    # tailles
    if "graph_size" in df.columns:
        df["graph_size"] = df["graph_size"].apply(_safe_float)

    # NB: certains CSV ont meta_alpha, meta_seed, etc.
    # on conserve algo + x + runtime + coûts disponibles
    keep = []
    for c in [args.x, "algorithm_name", "meta_alpha", "meta_seed", "seed", "flow", "runtime_sec"] + cost_cols:
        if c in df.columns and c not in keep:
            keep.append(c)

    if not keep:
        raise RuntimeError("No usable columns found in input CSV.")

    df = df[keep].copy()

    # Harmoniser la seed si besoin
    if "seed" not in df.columns and "meta_seed" in df.columns:
        df["seed"] = df["meta_seed"]
    if "meta_alpha" not in df.columns:
        df["meta_alpha"] = None

    group_cols = ["algorithm_name", args.by]
    if "meta_alpha" in df.columns:
        # utile si tu veux plusieurs courbes cost-aware alpha
        # on ne groupe par alpha que si au moins une valeur est non nulle/non NaN
        if df["meta_alpha"].notna().any():
            group_cols.append("meta_alpha")

    agg = {
        "runtime_sec": ["mean"]
    }
    for c in cost_cols:
        agg[c] = ["mean"]

    if args.include_std:
        agg["runtime_sec"].append("std")
        agg["runtime_sec"].append(_stderr)
        for c in cost_cols:
            agg[c].append("std")
            agg[c].append(_stderr)

    g = df.groupby(group_cols, dropna=False).agg(agg)

    # Flatten multi-index columns
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.reset_index()

    # Renommer proprement stderr
    if args.include_std:
        ren = {}
        for col in g.columns:
            if col.endswith("_<lambda>"):
                ren[col] = col.replace("_<lambda>", "_stderr")
        g = g.rename(columns=ren)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Plot table written to: {out_path}")


if __name__ == "__main__":
    main()
