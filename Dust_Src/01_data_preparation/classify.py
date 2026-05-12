#!/usr/bin/env python3
"""
Classify features into fast / medium / slow changing groups based on
how frequently they change over time for each node.

Usage:
  python classify_feature_timescales.py --csv GRID3km_DAILY_20x20_modelready_labeled_featured.csv

Output:
  - Prints a table with change_rate for each feature.
  - Prints lists of fast_feats / med_feats / slow_feats to copy into your GTG code.
"""

import argparse
import numpy as np
import pandas as pd


def load_daily_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"node", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {required}, missing: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    # IMPORTANT: sort once by node,date so groupby preserves time order
    df = df.sort_values(["node", "date"]).reset_index(drop=True)
    return df


def compute_change_rate_per_feature(df: pd.DataFrame, eps: float = 1e-6):
    """
    For each numeric feature column, compute a 'change_rate' in [0,1]:

      change_rate(feature) = mean over nodes of
         (#days with |x_t - x_{t-1}| > eps) / (#days - 1)

    Returns:
      pandas.DataFrame with columns:
        - feature
        - change_rate
        - node_mean_std  (mean over nodes of std over time)
        - global_std     (std over all node×time)
    """
    exclude_cols = {"node", "date", "label"}
    num_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    results = []

    # Pre-group by node once (already sorted by node,date)
    grouped = df.groupby("node", sort=False)

    for col in num_cols:
        per_node_change = []
        per_node_std = []

        for node, g in grouped:
            vals = g[col].values.astype(float)

            # Handle all-NaN or constant with NaNs
            if np.all(np.isnan(vals)):
                per_node_change.append(0.0)
                per_node_std.append(0.0)
                continue

            # Replace NaNs with node-wise mean to avoid spurious changes
            node_mean = np.nanmean(vals)
            vals = np.nan_to_num(vals, nan=node_mean)

            if len(vals) <= 1:
                per_node_change.append(0.0)
                per_node_std.append(0.0)
                continue

            diff = np.abs(np.diff(vals))
            changes = (diff > eps).astype(float)
            change_rate_node = changes.mean()  # fraction of days with a change
            per_node_change.append(change_rate_node)

            per_node_std.append(float(np.std(vals)))

        change_rate = float(np.mean(per_node_change)) if per_node_change else 0.0
        node_mean_std = float(np.mean(per_node_std)) if per_node_std else 0.0
        global_std = float(np.nanstd(df[col].values.astype(float)))

        results.append(
            {
                "feature": col,
                "change_rate": change_rate,
                "node_mean_std": node_mean_std,
                "global_std": global_std,
            }
        )

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("change_rate", ascending=False).reset_index(drop=True)
    return out_df


def classify_timescales(metrics_df: pd.DataFrame,
                        fast_thr: float = 0.30,
                        med_thr: float = 0.05):
    """
    Classify features into fast / medium / slow groups based on change_rate.

    Args:
      metrics_df: DataFrame from compute_change_rate_per_feature
      fast_thr:  change_rate >= fast_thr → fast
      med_thr:   med_thr <= change_rate < fast_thr → medium
                 change_rate < med_thr → slow

    Returns:
      dict with keys "fast", "medium", "slow" mapping to list of feature names.
    """
    fast_feats = metrics_df.loc[metrics_df["change_rate"] >= fast_thr, "feature"].tolist()
    med_mask = (metrics_df["change_rate"] >= med_thr) & (metrics_df["change_rate"] < fast_thr)
    med_feats = metrics_df.loc[med_mask, "feature"].tolist()
    slow_feats = metrics_df.loc[metrics_df["change_rate"] < med_thr, "feature"].tolist()

    return {
        "fast": fast_feats,
        "medium": med_feats,
        "slow": slow_feats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to daily CSV (with node,date,label + features)")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Change threshold; |x_t - x_{t-1}| > eps counts as a change")
    parser.add_argument("--fast-thr", type=float, default=0.30,
                        help="change_rate >= fast_thr → fast")
    parser.add_argument("--med-thr", type=float, default=0.05,
                        help="med_thr <= change_rate < fast_thr → medium")
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = load_daily_csv(args.csv)
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")

    print("Computing change_rate per feature...")
    metrics_df = compute_change_rate_per_feature(df, eps=args.eps)

    print("\n=== Feature temporal change metrics ===")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    groups = classify_timescales(metrics_df,
                                 fast_thr=args.fast_thr,
                                 med_thr=args.med_thr)

    print("\n=== Classified feature groups ===")
    print(f"\nFast-changing features (change_rate >= {args.fast_thr}):")
    print(groups["fast"])

    print(f"\nMedium-changing features ({args.med_thr} <= change_rate < {args.fast_thr}):")
    print(groups["medium"])

    print(f"\nSlow-changing features (change_rate < {args.med_thr}):")
    print(groups["slow"])

    print("\n=== Copy-paste for GTG code ===")
    print("fast_feats = [")
    for f in groups["fast"]:
        print(f'    "{f}",')
    print("]\n")

    print("med_feats = [")
    for f in groups["medium"]:
        print(f'    "{f}",')
    print("]\n")

    print("slow_feats = [")
    for f in groups["slow"]:
        print(f'    "{f}",')
    print("]")


if __name__ == "__main__":
    main()
