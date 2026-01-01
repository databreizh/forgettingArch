#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental Data Aggregator for Publication-Ready Tables.
Processes raw summary CSVs to compute mean performance metrics and 
statistical error bounds (std, stderr) for comparative plotting.
"""

import argparse
import math
from pathlib import Path
import pandas as pd


def _safe_float(x):
    """Safely converts input to float, returning None for invalid or missing values."""
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _stderr(series: pd.Series) -> float:
    """
    Computes the Standard Error of the Mean (SEM).
    Useful for plotting confidence intervals in comparative analysis.
    """
    s = series.dropna()
    n = len(s)
    if n <= 1:
        return float("nan")
    # ddof=1 for unbiased sample standard deviation
    return float(s.std(ddof=1) / math.sqrt(n))


def main():
    """
    Main transformation pipeline:
    Filters, groups, and aggregates metrics from large-scale experiment traces.
    """
    ap = argparse.ArgumentParser(
        description="Build compact CSV tables for plotting (runtime, cost) from aggregated summaries."
    )
    ap.add_argument("input_csv", help="Source aggregated CSV (e.g., summary_all.csv)")
    ap.add_argument("--out", default="experiments/plot_table_used_for_figures.csv",
                    help="Target CSV path for plot-ready data")
    ap.add_argument("--by", default="graph_size",
                    help="Grouping key for the X-axis (e.g., 'graph_size' or 'flow_name')")
    ap.add_argument("--x", default="graph_size",
                    help="Column name to retain as the independent variable (X-axis)")
    ap.add_argument("--include-std", action="store_true",
                    help="Compute and output standard deviation and standard error columns")
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.out)

    # Load experimental results
    df = pd.read_csv(in_path)

    # 1. Normalization of numerical columns
    # We ensure all metrics are treated as floats to prevent aggregation errors
    if "runtime_sec" in df.columns:
        df["runtime_sec"] = df["runtime_sec"].apply(_safe_float)

    # Supported cost metrics across various algorithm versions
    cost_cols = [c for c in [
        "weighted_cost", "raw_cost", "normalized_cost", 
        "relative_cost_vs_naive", "relative_cost"
    ] if c in df.columns]

    for c in cost_cols:
        df[c] = df[c].apply(_safe_float)

    if "graph_size" in df.columns:
        df["graph_size"] = df["graph_size"].apply(_safe_float)

    # 2. Column Filtering
    # Retain only necessary dimensions for visualization: Algorithm ID, X-axis, and Metrics
    keep = []
    candidates = [args.x, "algorithm_name", "meta_alpha", "meta_seed", "seed", "flow", "runtime_sec"] + cost_cols
    for c in candidates:
        if c in df.columns and c not in keep:
            keep.append(c)

    if not keep:
        raise RuntimeError(f"Data Integrity Error: No usable metrics found in {in_path}.")

    df = df[keep].copy()

    # Harmonize seed metadata for cross-version consistency
    if "seed" not in df.columns and "meta_seed" in df.columns:
        df["seed"] = df["meta_seed"]
    if "meta_alpha" not in df.columns:
        df["meta_alpha"] = None

    # 3. Statistical Aggregation
    # Group results by Algorithm and Independent Variable (e.g., Size or Flow)
    group_cols = ["algorithm_name", args.by]
    
    # Handle specific 'alpha' parameter for Cost-Aware variants
    if "meta_alpha" in df.columns and df["meta_alpha"].notna().any():
        group_cols.append("meta_alpha")

    # Define aggregation operations
    agg = {"runtime_sec": ["mean"]}
    for c in cost_cols:
        agg[c] = ["mean"]

    if args.include_std:
        # Add spread metrics for reliability analysis
        agg["runtime_sec"].extend(["std", _stderr])
        for c in cost_cols:
            agg[c].extend(["std", _stderr])

    # Compute statistics
    g = df.groupby(group_cols, dropna=False).agg(agg)

    # 4. Post-processing for CSV Export
    # Flatten Multi-index columns (e.g., ('runtime_sec', 'mean') -> 'runtime_sec_mean')
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.reset_index()

    # Clean up naming conventions for exported statistical headers
    if args.include_std:
        ren = {col: col.replace("_<lambda>", "_stderr") for col in g.columns if col.endswith("_<lambda>")}
        g = g.rename(columns=ren)

    # 5. Persistence
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Aggregated plot table generated: {out_path}")


if __name__ == "__main__":
    main()