"""
Visualization module for Collaborative Forgetting experiments.
Produces grouped bar plots comparing algorithms across different dataflows.
Optimized for academic publication (Grayscale/Camera-Ready support).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default paths for experiment results and figure exports
SUMMARY_PATH = "experiments_real/summary_real.csv"
FIG_DIR = "figures"


def ensure_dir(path: str) -> None:
    """
    Creates the target directory if it does not exist.
    Equivalent to 'mkdir -p' in Unix systems.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def grouped_bar_plot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    outfile: str,
    logy: bool = False,
) -> None:
    """
    Generates a grouped bar chart for performance comparison.
    
    Args:
        df: Input DataFrame containing experiment results.
        metric: The column name to be plotted on the Y-axis.
        ylabel: Legend label for the Y-axis.
        outfile: Output filename for the generated figure.
        logy: Enables logarithmic scale for wide-range distributions.
    """
    # Compute the mean value per (flow, algorithm) pair
    g = (
        df.groupby(["flow_name", "algorithm_name"], as_index=False)[metric]
        .mean()
    )

    flows = sorted(g["flow_name"].unique())
    algos = sorted(g["algorithm_name"].unique())

    # Reshape data: index = flow_name, columns = algorithm_name
    pivot = g.pivot(index="flow_name", columns="algorithm_name", values=metric)
    pivot = pivot.reindex(flows)

    x = np.arange(len(flows))
    width = 0.8 / max(1, len(algos))

    fig, ax = plt.subplots(figsize=(6, 4))

    # Grayscale styles for Camera-Ready publications (ensures readability in B&W)
    ALGO_STYLE = {
        "greedy": {
            "color": "0.7",   # Light gray
            "hatch": "//",
        },
        "cost_aware": {
            "color": "0.4",   # Medium gray
            "hatch": "xx",
        },
        "cluster_based": {
            "color": "0.0",   # Black
            "hatch": "..",
        },
    }

    # Iterate through algorithms to plot grouped bars
    for i, algo in enumerate(algos):
        if algo not in pivot.columns:
            continue
        
        style = ALGO_STYLE.get(algo, {})
        values = pivot[algo].values
        
        # Calculate horizontal offset to group bars around the x-tick
        offset = (i - (len(algos) - 1) / 2) * width
        
        ax.bar(
            x + offset,
            values,
            width,
            label=algo,
            color=style.get("color", "0.5"),
            hatch=style.get("hatch", None),
            edgecolor="black",
        )

    # Labeling and aesthetic formatting
    ax.set_xticks(x)
    ax.set_xticklabels(flows)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Flow Name")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Logarithmic scale for metrics with significant trade-offs (e.g., Runtime or Cost)
    if logy:
        ax.set_yscale("log")

    plt.tight_layout()
    ensure_dir(FIG_DIR)
    
    out_path = os.path.join(FIG_DIR, outfile)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Figure generated: {metric} -> {out_path}")


def main() -> None:
    """
    Main execution pipeline: data loading, preprocessing, and plot generation.
    """
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Please ensure 'run_real_flows.py' "
            "and 'aggregate_real_summaries.py' have been executed."
        )

    df = pd.read_csv(SUMMARY_PATH)
    
    import re

    # ---------------------------------------------------------------------
    # Backward-compatibility: Infer flow_name from metadata if missing
    # ---------------------------------------------------------------------
    if "flow_name" not in df.columns:
        def infer_flow(row):
            # 1) Search in source_file path: experiments_real/<flow_name>/...
            sf = str(row.get("source_file", ""))
            m = re.search(r"experiments_real[\\/](\w+)[\\/]", sf)
            if m:
                return m.group(1)

            # 2) Search in experiment string: <flow>_seedX_algo
            exp = str(row.get("experiment", ""))
            m = re.match(r"^([a-zA-Z0-9]+)_seed\d+_", exp)
            if m:
                return m.group(1)

            return "unknown"

        df["flow_name"] = df.apply(infer_flow, axis=1)
        print("[INFO] 'flow_name' column was missing; inferred from metadata.")

    # Validation: Ensure mandatory columns for comparative analysis exist
    required_cols = ["flow_name", "algorithm_name", "deleted_size", "runtime_sec"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory column: {col} in {SUMMARY_PATH}")

    # ---------------------------------------------------------------------
    # Figure Generation Phase
    # ---------------------------------------------------------------------

    # 1) Deletion Impact (Linear scale)
    grouped_bar_plot(
        df=df,
        metric="deleted_size",
        ylabel="Number of Deleted Nodes",
        outfile="real_flows_deleted_size_bar.png",
        logy=False,
    )

    # 2) Weighted Deletion Cost (Log scale to handle heterogeneous costs)
    if "weighted_cost" in df.columns:
        grouped_bar_plot(
            df=df,
            metric="weighted_cost",
            ylabel="Weighted Deletion Cost (log scale)",
            outfile="real_flows_weighted_cost_bar_log.png",
            logy=True,
        )
    else:
        print("[WARN] 'weighted_cost' column missing; skipping cost visualization.")

    # 3) Computational Runtime (Log scale for scalability analysis)
    grouped_bar_plot(
        df=df,
        metric="runtime_sec",
        ylabel="Execution Runtime (seconds)",
        outfile="real_flows_runtime_bar_log.png",
        logy=True,
    )


if __name__ == "__main__":
    main()