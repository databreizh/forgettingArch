import os
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
SUMMARY_CSV = os.path.join("experiments", "summary_all.csv")
FIG_DIR = "figures"
PLOT_TABLE_CSV = os.path.join("experiments", "plot_table_used_for_figures.csv")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# Camera-ready B/W styles (line + marker). No explicit colors.
BW_STYLES: Dict[str, Dict[str, Any]] = {
    "greedy": dict(linestyle="--", marker="o", linewidth=2, markersize=5),
    "cost_aware": dict(linestyle="-", marker="^", linewidth=2, markersize=5),
    "cluster_based": dict(linestyle=":", marker="s", linewidth=2, markersize=5),
    # Option A v2 (alphas)
    "cost_aware_a02": dict(linestyle="-", marker="^", linewidth=2, markersize=5),
    "cost_aware_a05": dict(linestyle="-.", marker="^", linewidth=2, markersize=5),
    "cost_aware_a08": dict(linestyle=":", marker="^", linewidth=2, markersize=5),
    # baselines éventuels
    "naive": dict(linestyle="--", marker="x", linewidth=2, markersize=5),
    "exact": dict(linestyle="-", marker="*", linewidth=2, markersize=6),
}


def style_for(algo: str) -> Dict[str, Any]:
    return BW_STYLES.get(algo, dict(linestyle="-", marker="o", linewidth=2, markersize=5))


def apply_bw_axes(ax) -> None:
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, ncol=2)


def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def aggregate_for_plot(
    df: pd.DataFrame,
    y_col: str,
    group_cols: Tuple[str, str] = ("algorithm_name", "graph_size"),
) -> pd.DataFrame:
    """
    Returns mean(y_col) per (algorithm_name, graph_size).
    """
    if y_col not in df.columns:
        return pd.DataFrame(columns=list(group_cols) + [y_col])

    tmp = df.copy()
    _coerce_numeric(tmp, y_col)
    _coerce_numeric(tmp, "graph_size")

    out = (
        tmp.groupby(list(group_cols))[y_col]
        .mean()
        .reset_index()
        .sort_values(["algorithm_name", "graph_size"])
    )
    return out


def save_plot_table(
    runtime_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    out_csv: str,
) -> None:
    """
    Merge the aggregated tables to produce the exact values used for plotting.
    """
    ensure_dir(os.path.dirname(out_csv) or ".")
    merged = runtime_df.rename(columns={"runtime_sec": "runtime_mean_sec"}).copy()

    if not rel_df.empty and rel_df.columns.tolist()[-1] in ("relative_cost_vs_naive", "relative_cost"):
        col = rel_df.columns.tolist()[-1]
        merged = merged.merge(
            rel_df.rename(columns={col: f"{col}_mean"}),
            on=["algorithm_name", "graph_size"],
            how="left",
        )

    if not norm_df.empty and "normalized_cost" in norm_df.columns:
        merged = merged.merge(
            norm_df.rename(columns={"normalized_cost": "normalized_cost_mean"}),
            on=["algorithm_name", "graph_size"],
            how="left",
        )

    merged.to_csv(out_csv, index=False)
    print(f"[OK] Plot table (values used for figures): {out_csv}")


def plot_line_by_algo(
    df_agg: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    out_png: str,
    out_pdf: Optional[str] = None,
    yscale: Optional[str] = None,
    xscale: Optional[str] = "log",   # ✅ option A : log par défaut
) -> None:
    """
    Plot mean(y_col) vs x_col for each algorithm_name using true B/W rendering.
    """
    ensure_dir(os.path.dirname(out_png) or ".")
    if out_pdf:
        ensure_dir(os.path.dirname(out_pdf) or ".")

    fig, ax = plt.subplots()

    if df_agg.empty or y_col not in df_agg.columns or x_col not in df_agg.columns:
        print(f"[WARN] No data to plot for {y_col}.")
        plt.close(fig)
        return

    # ✅ Option A: log scale on x-axis (graph sizes)
    if xscale in ("log", "symlog"):
        # log requires strictly positive x
        df_plot = df_agg[df_agg[x_col] > 0].copy()
        if df_plot.empty:
            print(f"[WARN] Cannot use xscale={xscale}: no positive values in {x_col}.")
            df_plot = df_agg.copy()
            xscale = None
        else:
            df_agg = df_plot
            ax.set_xscale(xscale)

    # Force deterministic algo ordering (optional but nicer)
    for algo in sorted(df_agg["algorithm_name"].unique()):
        g = df_agg[df_agg["algorithm_name"] == algo].sort_values(x_col)
        if g[y_col].dropna().empty:
            continue
        st = style_for(algo)

        # True B/W: single black line, hollow markers
        ax.plot(
            g[x_col],
            g[y_col],
            label=algo,
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=st["linewidth"],
            markersize=st["markersize"],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yscale in ("log", "symlog"):
        ax.set_yscale(yscale)

    apply_bw_axes(ax)
    fig.tight_layout()

    fig.savefig(out_png, dpi=300)
    print(f"[OK] Figure: {out_png}")

    if out_pdf:
        fig.savefig(out_pdf)
        print(f"[OK] Figure: {out_pdf}")

    plt.close(fig)



# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    ensure_dir(FIG_DIR)
    ensure_dir(os.path.dirname(PLOT_TABLE_CSV) or ".")

    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(
            f"CSV agrégé introuvable : {SUMMARY_CSV}. "
            "Lance d'abord : python cfg\\experiments\\aggregate_summaries.py <input_dir> "
            f"--output {SUMMARY_CSV}"
        )

    df = pd.read_csv(SUMMARY_CSV)

    # Coerce key columns
    _coerce_numeric(df, "graph_size")
    _coerce_numeric(df, "runtime_sec")
    _coerce_numeric(df, "raw_cost")
    _coerce_numeric(df, "normalized_cost")

    # ------------------------------------------------------------
    # Relative cost vs Naive (robust): compute a baseline per run key then ratio
    # Creates df["relative_cost_vs_naive"] (may contain NA if no matching naive)
    # ------------------------------------------------------------
    rel_col = "relative_cost_vs_naive"
    df[rel_col] = pd.NA

    if "raw_cost" not in df.columns:
        print("[WARN] raw_cost absent -> impossible de calculer un coût relatif vs naive.")
    else:
        # Choose join keys: try to match the *same run conditions*
        join_keys = ["graph_size"]
        if "meta_seed" in df.columns:
            join_keys.append("meta_seed")
        elif "seed" in df.columns:
            join_keys.append("seed")

        if "initial_size" in df.columns:
            join_keys.append("initial_size")

        # If initial_nodes exists (even as a string), it's the strictest match
        if "initial_nodes" in df.columns:
            join_keys.append("initial_nodes")

        naive_df = df[df["algorithm_name"].astype(str).str.lower() == "naive"].copy()
        if naive_df.empty:
            print("[WARN] Aucun run naive dans le CSV -> pas de relative_cost_vs_naive.")
        else:
            naive_base = (
                naive_df.groupby(join_keys, dropna=False)["raw_cost"]
                .mean()
                .reset_index()
                .rename(columns={"raw_cost": "naive_raw_cost"})
            )

            df = df.merge(naive_base, on=join_keys, how="left")
            # Ancien calcul direct -> remplacé par garde-fou
            if "naive_raw_cost" in df.columns:
                mask = df["raw_cost"].notna() & df["naive_raw_cost"].notna() & (df["naive_raw_cost"] > 0)
                df.loc[mask, "relative_cost_vs_naive"] = df.loc[mask, "raw_cost"] / df.loc[mask, "naive_raw_cost"]
            else:
                print("[WARN] naive_raw_cost absent -> relative_cost_vs_naive non calculé (pas de baseline naive joignable).")




    # -------------------------
    # Runtime vs size
    # -------------------------
    runtime_agg = aggregate_for_plot(df, "runtime_sec")
    plot_line_by_algo(
        df_agg=runtime_agg,
        x_col="graph_size",
        y_col="runtime_sec",
        xlabel="Graph size (nodes)",
        ylabel="Runtime (sec)",
        out_png=os.path.join(FIG_DIR, "runtime_vs_size.png"),
        out_pdf=os.path.join(FIG_DIR, "runtime_vs_size.pdf"),
        yscale=None,
    )

    # -------------------------
    # Normalized cost vs size
    # -------------------------
    norm_agg = pd.DataFrame()
    if "normalized_cost" in df.columns and not df["normalized_cost"].dropna().empty:
        norm_agg = aggregate_for_plot(df, "normalized_cost")
        plot_line_by_algo(
            df_agg=norm_agg,
            x_col="graph_size",
            y_col="normalized_cost",
            xlabel="Graph size (nodes)",
            ylabel="Normalized cost",
            out_png=os.path.join(FIG_DIR, "normalized_cost_vs_size.png"),
            out_pdf=os.path.join(FIG_DIR, "normalized_cost_vs_size.pdf"),
            yscale=None,
        )
    else:
        print("[WARN] normalized_cost absent ou vide -> pas de figure normalized_cost_vs_size.")

    # -------------------------
    # Relative cost vs size (vs Naive)
    # -------------------------
    rel_agg = pd.DataFrame()
    if rel_col and rel_col in df.columns and not df[rel_col].dropna().empty:
        rel_agg = aggregate_for_plot(df, rel_col)
        plot_line_by_algo(
            df_agg=rel_agg,
            x_col="graph_size",
            y_col=rel_col,
            xlabel="Graph size (nodes)",
            ylabel="Relative cost (vs Naive)",
            out_png=os.path.join(FIG_DIR, "relative_cost_vs_size.png"),
            out_pdf=os.path.join(FIG_DIR, "relative_cost_vs_size.pdf"),
            yscale=None,
        )
    else:
        print("[WARN] relative_cost_vs_naive absent ou vide -> pas de figure relative_cost_vs_size.")

    # -------------------------
    # Save plot table (values used)
    # -------------------------
    save_plot_table(runtime_agg, rel_agg, norm_agg, PLOT_TABLE_CSV)


if __name__ == "__main__":
    main()
