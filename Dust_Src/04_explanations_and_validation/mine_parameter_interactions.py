#!/usr/bin/env python3
"""
Mine feature-lag interaction patterns from mapped GTG explanations.

Input:
  - Flattened mapped explanations CSV (e.g., gtg_s77_rich_mapped_explanations_flat.csv)

Outputs:
  - per_event_pairs.csv
  - interaction_pairs_ranked.csv
  - interaction_pairs_shortlist.csv
  - feature_lag_stats.csv
  - summary.json

This is an improved, direct-file version of the earlier shell+inline-python workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "feature",
    "lag_days",
    "contribution",
]


def parse_lag_bins(raw: str) -> List[int]:
    bins = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not bins:
        raise ValueError("lag bins are empty")
    if bins[0] != 0:
        raise ValueError(f"lag bins must start with 0, got {bins[0]}")
    if bins != sorted(bins):
        raise ValueError(f"lag bins must be sorted ascending, got {bins}")
    if len(set(bins)) != len(bins):
        raise ValueError(f"lag bins contain duplicates, got {bins}")
    return bins


def lag_bin_labels(bins: List[int]) -> tuple[list[float], list[str]]:
    # Add an overflow bucket so very long lags are still categorized.
    edges = [float(x) for x in bins] + [np.inf]
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)] + [f">{bins[-1]}"]
    return edges, labels


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    # Handles 0/1, true/false-like strings.
    return s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def pick_first_existing(cols: Iterable[str], available: set[str]) -> str | None:
    for c in cols:
        if c in available:
            return c
    return None


def mine_interactions(
    df: pd.DataFrame,
    event_col: str,
    date_col: str,
    prob_col: str,
    label_col: str | None,
    topk: int,
    lag_bins: List[int],
    positive_only: bool,
    positive_threshold: float,
    drop_missing_in_csv: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy()

    # Type coercions.
    df["lag_days"] = pd.to_numeric(df["lag_days"], errors="coerce")
    df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce")
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df[date_col] = safe_to_datetime(df[date_col])

    if label_col is not None:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    if "is_anomalous" in df.columns:
        df["is_anomalous"] = as_bool_series(df["is_anomalous"])
    else:
        df["is_anomalous"] = False

    if drop_missing_in_csv and "missing_in_csv" in df.columns:
        miss_mask = as_bool_series(df["missing_in_csv"])
        df = df.loc[~miss_mask].copy()

    # Required non-null.
    core_cols = [event_col, "feature", "lag_days", "contribution", prob_col, date_col]
    df = df.dropna(subset=core_cols).copy()

    # Label filter.
    if positive_only and label_col is not None:
        df = df.loc[df[label_col] > positive_threshold].copy()

    # Keep sensible lags only.
    df = df.loc[df["lag_days"] >= 0].copy()

    if df.empty:
        empty = pd.DataFrame()
        summary = {
            "warning": "No rows after filtering. Check label filter, thresholds, and input columns."
        }
        return empty, empty, empty, empty, summary

    edges, labels = lag_bin_labels(lag_bins)
    df["lag_bin"] = pd.cut(
        df["lag_days"].astype(float),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(str)
    df["abs_contrib"] = df["contribution"].abs()

    # Keep strongest row per (event, feature) to avoid feature duplication per event.
    df_best = (
        df.sort_values([event_col, "abs_contrib"], ascending=[True, False])
        .drop_duplicates(subset=[event_col, "feature"], keep="first")
        .copy()
    )

    # Keep top-k strongest features per event.
    df_top = (
        df_best.sort_values([event_col, "abs_contrib"], ascending=[True, False])
        .groupby(event_col, as_index=False)
        .head(topk)
        .copy()
    )

    # Event metadata.
    meta_cols = [event_col, date_col, prob_col]
    if label_col is not None:
        meta_cols.append(label_col)
    for c in ["node", "longitude", "latitude", "time_index", "local_node"]:
        if c in df.columns:
            meta_cols.append(c)
    event_meta = (
        df[meta_cols]
        .sort_values([event_col, date_col])
        .drop_duplicates(subset=[event_col], keep="last")
        .set_index(event_col)
    )

    # Per-event pair generation.
    pairs = []
    for event_id, g in df_top.groupby(event_col):
        rows = g.to_dict("records")
        if len(rows) < 2:
            continue

        meta = event_meta.loc[event_id]
        for a, b in combinations(rows, 2):
            # Canonical pair key for stable aggregation.
            ka = (str(a["feature"]), str(a["lag_bin"]))
            kb = (str(b["feature"]), str(b["lag_bin"]))
            (fa, la), (fb, lb) = sorted([ka, kb], key=lambda x: (x[0], x[1]))

            ga = str(a.get("group", ""))
            gb = str(b.get("group", ""))
            rec = {
                "event_id": event_id,
                "event_date": meta[date_col].date().isoformat() if pd.notna(meta[date_col]) else None,
                "prob": float(meta[prob_col]) if pd.notna(meta[prob_col]) else np.nan,
                "feature_a": fa,
                "lagbin_a": la,
                "feature_b": fb,
                "lagbin_b": lb,
                "sum_abs_contrib": float(a["abs_contrib"] + b["abs_contrib"]),
                "mean_abs_contrib_pair": float(0.5 * (a["abs_contrib"] + b["abs_contrib"])),
                "both_anomalous": bool(a.get("is_anomalous", False) and b.get("is_anomalous", False)),
                "same_group": bool(ga == gb) if (ga and gb) else False,
                "groups": ",".join(sorted([ga, gb])),
            }
            if label_col is not None:
                rec["label"] = float(meta[label_col]) if pd.notna(meta[label_col]) else np.nan
            for c in ["node", "longitude", "latitude", "time_index", "local_node"]:
                if c in meta.index and pd.notna(meta[c]):
                    rec[c] = float(meta[c]) if c in {"longitude", "latitude"} else meta[c]
            pairs.append(rec)

    pairs_df = pd.DataFrame(pairs)
    n_events = int(df_top[event_col].nunique())

    if pairs_df.empty:
        empty = pd.DataFrame()
        summary = {
            "warning": "No pair records were created (topk may be too small or events have <2 valid features).",
            "counts": {
                "unique_events": n_events,
                "rows_filtered": int(len(df)),
                "rows_best_per_event_feature": int(len(df_best)),
                "rows_topk": int(len(df_top)),
                "pairs_total": 0,
            },
        }
        return pairs_df, empty, empty, empty, summary

    # Pair-level aggregation.
    group_cols = ["feature_a", "lagbin_a", "feature_b", "lagbin_b"]
    agg = (
        pairs_df.groupby(group_cols, as_index=False)
        .agg(
            count_events=("event_id", "nunique"),
            mean_prob=("prob", "mean"),
            mean_sum_abs_contrib=("sum_abs_contrib", "mean"),
            median_sum_abs_contrib=("sum_abs_contrib", "median"),
            both_anomalous_rate=("both_anomalous", "mean"),
            same_group_rate=("same_group", "mean"),
        )
    )
    if "label" in pairs_df.columns:
        agg["pos_label_rate"] = (
            pairs_df.groupby(group_cols)["label"]
            .mean()
            .reset_index(drop=True)
        )
    else:
        agg["pos_label_rate"] = np.nan

    agg["support"] = agg["count_events"] / max(n_events, 1)

    # Marginal item supports for lift.
    marg = (
        df_top[[event_col, "feature", "lag_bin"]]
        .drop_duplicates()
        .groupby(["feature", "lag_bin"], as_index=False)
        .agg(item_count=(event_col, "nunique"))
    )
    marg["item_support"] = marg["item_count"] / max(n_events, 1)

    marg_a = marg.rename(
        columns={
            "feature": "feature_a",
            "lag_bin": "lagbin_a",
            "item_count": "item_count_a",
            "item_support": "item_support_a",
        }
    )
    marg_b = marg.rename(
        columns={
            "feature": "feature_b",
            "lag_bin": "lagbin_b",
            "item_count": "item_count_b",
            "item_support": "item_support_b",
        }
    )
    agg = agg.merge(marg_a, on=["feature_a", "lagbin_a"], how="left")
    agg = agg.merge(marg_b, on=["feature_b", "lagbin_b"], how="left")
    agg["expected_support_indep"] = agg["item_support_a"] * agg["item_support_b"]
    agg["lift"] = np.where(
        agg["expected_support_indep"] > 0,
        agg["support"] / agg["expected_support_indep"],
        np.nan,
    )
    agg["pmi_log2"] = np.log2(agg["lift"])

    # Composite score: stable frequency + strength + association.
    lift_clip = agg["lift"].clip(lower=0.25, upper=8.0).fillna(1.0)
    agg["pair_score"] = (
        np.log1p(agg["count_events"]) * agg["mean_sum_abs_contrib"] * np.sqrt(lift_clip)
    )
    agg = agg.sort_values(
        ["pair_score", "count_events", "mean_sum_abs_contrib"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Feature-level lag statistics.
    lag_stats = (
        df_best.groupby("feature", as_index=False)
        .agg(
            n_rows=(event_col, "count"),
            n_events=(event_col, "nunique"),
            mean_abs_contrib=("abs_contrib", "mean"),
            median_abs_contrib=("abs_contrib", "median"),
            median_lag=("lag_days", "median"),
            p90_lag=("lag_days", lambda x: float(np.percentile(x, 90))),
            p99_lag=("lag_days", lambda x: float(np.percentile(x, 99))),
            anomalous_rate=("is_anomalous", "mean"),
        )
        .sort_values(["mean_abs_contrib", "n_events"], ascending=[False, False])
        .reset_index(drop=True)
    )

    summary = {
        "counts": {
            "rows_input_after_filtering": int(len(df)),
            "unique_events": n_events,
            "rows_best_per_event_feature": int(len(df_best)),
            "rows_topk": int(len(df_top)),
            "pairs_total": int(len(pairs_df)),
            "unique_pair_patterns": int(len(agg)),
        }
    }

    return pairs_df, agg, lag_stats, df_top, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine feature-lag interaction pairs from mapped explanations.")
    parser.add_argument("--csv", type=Path, required=True, help="Mapped explanations flat CSV path.")
    parser.add_argument("--outdir", type=Path, default=Path("analysis_out"), help="Output directory.")
    parser.add_argument("--topk", type=int, default=8, help="Top-k strongest features per event before pairing.")
    parser.add_argument(
        "--lag-bins",
        type=str,
        default="0,7,30,90,180,365,730,1095,100000",
        help="Comma-separated lag bin edges (must start at 0).",
    )
    parser.add_argument("--event-col", type=str, default="nid", help="Event identifier column.")
    parser.add_argument("--date-col", type=str, default="event_date", help="Event date column.")
    parser.add_argument("--prob-col", type=str, default="prob", help="Probability column.")
    parser.add_argument("--label-col", type=str, default="label", help="Label column (set empty to disable).")
    parser.add_argument(
        "--positive-only",
        action="store_true",
        default=True,
        help="Keep only rows with label > positive-threshold (if label column exists).",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.5,
        help="Threshold for positive labels when --positive-only is enabled.",
    )
    parser.add_argument(
        "--drop-missing-in-csv",
        action="store_true",
        default=True,
        help="Drop explanation rows where missing_in_csv is true (if column exists).",
    )
    parser.add_argument(
        "--min-support-frac",
        type=float,
        default=0.03,
        help="Shortlist minimum support as fraction of unique events.",
    )
    parser.add_argument(
        "--min-support-count",
        type=int,
        default=2,
        help="Shortlist minimum support as absolute event count.",
    )
    args = parser.parse_args()

    if args.topk < 2:
        raise ValueError("--topk must be >= 2 to form pairs")

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    lag_bins = parse_lag_bins(args.lag_bins)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    cols = set(df.columns)

    missing_req = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing_req:
        raise ValueError(f"Input CSV missing required columns: {missing_req}")

    for c in [args.event_col, args.date_col, args.prob_col]:
        if c not in cols:
            raise ValueError(f"Input CSV missing required column: {c}")

    label_col = args.label_col.strip() if args.label_col else None
    if label_col and label_col not in cols:
        print(f"[WARN] label column '{label_col}' not found; disabling label-based filtering.")
        label_col = None

    pairs_df, agg, lag_stats, df_top, summary = mine_interactions(
        df=df,
        event_col=args.event_col,
        date_col=args.date_col,
        prob_col=args.prob_col,
        label_col=label_col,
        topk=args.topk,
        lag_bins=lag_bins,
        positive_only=args.positive_only,
        positive_threshold=args.positive_threshold,
        drop_missing_in_csv=args.drop_missing_in_csv,
    )

    # Save outputs.
    per_event_pairs_csv = outdir / "per_event_pairs.csv"
    ranked_csv = outdir / "interaction_pairs_ranked.csv"
    shortlist_csv = outdir / "interaction_pairs_shortlist.csv"
    lag_stats_csv = outdir / "feature_lag_stats.csv"
    topk_rows_csv = outdir / "topk_rows_per_event.csv"
    summary_json = outdir / "summary.json"

    pairs_df.to_csv(per_event_pairs_csv, index=False)
    agg.to_csv(ranked_csv, index=False)
    lag_stats.to_csv(lag_stats_csv, index=False)
    df_top.to_csv(topk_rows_csv, index=False)

    n_events = int(summary.get("counts", {}).get("unique_events", 0))
    min_support = max(args.min_support_count, int(math.ceil(args.min_support_frac * max(n_events, 1))))
    shortlist = agg.loc[agg["count_events"] >= min_support].copy() if not agg.empty else agg.copy()
    shortlist = shortlist.sort_values(
        ["pair_score", "count_events", "mean_sum_abs_contrib"],
        ascending=[False, False, False],
    )
    shortlist.to_csv(shortlist_csv, index=False)

    summary.update(
        {
            "inputs": {
                "csv": str(args.csv),
                "event_col": args.event_col,
                "date_col": args.date_col,
                "prob_col": args.prob_col,
                "label_col": label_col,
                "topk": args.topk,
                "lag_bins_days": lag_bins,
                "positive_only": bool(args.positive_only),
                "positive_threshold": float(args.positive_threshold),
                "drop_missing_in_csv": bool(args.drop_missing_in_csv),
                "min_support_frac": float(args.min_support_frac),
                "min_support_count": int(args.min_support_count),
                "shortlist_min_events": int(min_support),
            },
            "outputs": {
                "per_event_pairs.csv": str(per_event_pairs_csv),
                "interaction_pairs_ranked.csv": str(ranked_csv),
                "interaction_pairs_shortlist.csv": str(shortlist_csv),
                "feature_lag_stats.csv": str(lag_stats_csv),
                "topk_rows_per_event.csv": str(topk_rows_csv),
            },
            "counts": {
                **summary.get("counts", {}),
                "shortlist_rows": int(len(shortlist)),
            },
        }
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] wrote {per_event_pairs_csv}")
    print(f"[OK] wrote {ranked_csv}")
    print(f"[OK] wrote {shortlist_csv}")
    print(f"[OK] wrote {lag_stats_csv}")
    print(f"[OK] wrote {topk_rows_csv}")
    print(f"[OK] wrote {summary_json}")

    if not agg.empty:
        view_cols = [
            "feature_a",
            "lagbin_a",
            "feature_b",
            "lagbin_b",
            "count_events",
            "support",
            "mean_sum_abs_contrib",
            "lift",
            "pair_score",
        ]
        show_cols = [c for c in view_cols if c in agg.columns]
        print("\n=== Top interaction pair patterns (first 15) ===")
        print(agg.loc[:, show_cols].head(15).to_string(index=False))
    else:
        print("\n[WARN] No aggregated pair patterns found.")


if __name__ == "__main__":
    main()

