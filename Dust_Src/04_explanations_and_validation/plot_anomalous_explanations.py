#!/usr/bin/env python3
"""plot_anomalous_explanations.py

Create per-explanation PNG figures that show each (top) feature time series
around its lag neighborhood.

Neighborhood definition (matches your request):
For a feature with lag_days = L (relative to event_date),
we plot the feature from (L-window) to (L+window) days of lag,
i.e. calendar dates:
  start = event_date - (L + window) days
  end   = event_date - (L - window) days

So example: lag_days=800, window=50 => plot lag 750..850 days.

Inputs:
- mapped_explanations.json (output from map_explanations.py) OR any explanation JSON list
  that contains: local_node/node/event date/top_features[*].feature/lag_days/lag_date/is_anomalous.
- modelready CSV containing per-day rows: node,date,feature columns.

Outputs:
- One PNG per selected explanation into --out_dir.

Usage:
  python3 plot_anomalous_explanations.py \
    --data GRID3km_DAILY_20x20_modelready_labeled_featured.csv \
    --mapped mapped_explanations.json \
    --out_dir plots \
    --window 50 \
    --zthr 2.0 \
    --max_events 50 \
    --require_all_anom

"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def norm_date(s: Any) -> str:
    return str(s)[:10] if s is not None else ""


def load_mapped(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("mapped JSON must be a list")
    return data


def select_events(
    exps: List[Dict[str, Any]],
    require_all_anom: bool,
    min_anom: int,
    max_events: int,
) -> List[Dict[str, Any]]:
    scored = []
    for e in exps:
        tfs = [tf for tf in e.get("top_features", []) if not tf.get("missing_in_csv", False)]
        if not tfs:
            continue
        n_anom = sum(1 for tf in tfs if bool(tf.get("is_anomalous", False)))
        ok = (n_anom >= min_anom)
        if require_all_anom:
            ok = ok and (n_anom == len(tfs))
        if ok:
            # sort key: (prob desc, then count anomalous desc)
            prob = float(e.get("prob", 0.0) or 0.0)
            scored.append((prob, n_anom, e))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored[:max_events]]


def stream_load_nodes(
    data_csv: Path,
    nodes_needed: Set[str],
    feats_needed: Set[str],
    chunksize: int = 500_000,
) -> Dict[str, pd.DataFrame]:
    """Stream CSV once and keep only rows for nodes_needed, with required feature columns."""
    header = pd.read_csv(data_csv, nrows=0).columns.tolist()
    header_set = set(header)

    feats_avail = sorted([f for f in feats_needed if f in header_set])
    missing = sorted([f for f in feats_needed if f not in header_set])
    if missing:
        print(f"[WARN] {len(missing)} features missing in CSV (will be skipped in plots). Example: {missing[:10]}")

    usecols = ["node", "date"] + feats_avail
    per_node_parts: Dict[str, List[pd.DataFrame]] = {n: [] for n in nodes_needed}

    for chunk in pd.read_csv(data_csv, usecols=usecols, chunksize=chunksize):
        chunk["node"] = chunk["node"].astype(str)
        m = chunk["node"].isin(nodes_needed)
        if not m.any():
            continue
        chunk = chunk.loc[m].copy()
        # normalize date string now; convert to datetime later after concat
        chunk["date"] = chunk["date"].astype(str).str.slice(0, 10)
        for n, df_n in chunk.groupby("node"):
            per_node_parts[str(n)].append(df_n)

    per_node: Dict[str, pd.DataFrame] = {}
    for n, parts in per_node_parts.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        per_node[n] = df

    return per_node


def plot_event(
    e: Dict[str, Any],
    df_node: pd.DataFrame,
    out_path: Path,
    window: int,
    zthr: float,
):
    node = str(e.get("node", ""))
    nid = e.get("nid", "")
    event_date = pd.to_datetime(norm_date(e.get("date")), errors="coerce")
    prob = e.get("prob", None)

    tfs = [tf for tf in e.get("top_features", []) if not tf.get("missing_in_csv", False)]
    if not tfs or pd.isna(event_date):
        return

    n = len(tfs)
    fig_h = max(2.5 * n, 3.0)
    fig = plt.figure(figsize=(14, fig_h), dpi=150)

    for k, tf in enumerate(tfs, start=1):
        feat = tf["feature"]
        lag_days = int(tf.get("lag_days", 0) or 0)
        lag_date = pd.to_datetime(norm_date(tf.get("lag_date")), errors="coerce")

        # neighborhood in calendar time based on lag offset
        start = event_date - pd.Timedelta(days=lag_days + window)
        end   = event_date - pd.Timedelta(days=max(lag_days - window, 0))

        ax = fig.add_subplot(n, 1, k)

        if feat not in df_node.columns:
            ax.text(0.5, 0.5, f"Missing feature column: {feat}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        seg = df_node[(df_node["date"] >= start) & (df_node["date"] <= end)][["date", feat]].copy()
        ax.plot(seg["date"], seg[feat], linewidth=1.4)

        # mark lag date
        if not pd.isna(lag_date):
            ax.axvline(lag_date, linestyle="--", linewidth=1.0)
            # mark actual value if present
            row = df_node[df_node["date"] == lag_date]
            if len(row):
                try:
                    y = float(row.iloc[-1][feat])
                    ax.scatter([lag_date], [y], s=18)
                except Exception:
                    pass

        # optional anomaly threshold lines (node_mean +/- zthr*node_std) if present
        mu = tf.get("node_mean", None)
        sd = tf.get("node_std", None)
        if (mu is not None) and (sd is not None):
            try:
                mu = float(mu); sd = float(sd)
                ax.axhline(mu + zthr * sd, linestyle=":", linewidth=1.0)
                ax.axhline(mu - zthr * sd, linestyle=":", linewidth=1.0)
            except Exception:
                pass

        z = tf.get("zscore", None)
        is_anom = bool(tf.get("is_anomalous", False))
        ax.set_title(f"{feat} | lag_days={lag_days} | z={None if z is None else round(float(z),2)} | anomalous={is_anom}")
        ax.grid(True, alpha=0.25)
        if k < n:
            ax.set_xlabel("")

    title = f"nid={nid} | node={node} | event_date={norm_date(e.get('date'))} | prob={prob}"
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="modelready CSV")
    ap.add_argument("--mapped", required=True, type=Path, help="mapped_explanations.json")
    ap.add_argument("--out_dir", required=True, type=Path, help="output directory for PNGs")
    ap.add_argument("--window", default=50, type=int, help="lag neighborhood half-width (days)")
    ap.add_argument("--zthr", default=2.0, type=float, help="z-threshold used for guide lines")
    ap.add_argument("--chunksize", default=500_000, type=int)
    ap.add_argument("--max_events", default=50, type=int)
    ap.add_argument("--require_all_anom", action="store_true",
                    help="only plot explanations where ALL non-missing top_features are anomalous")
    ap.add_argument("--min_anom", default=1, type=int,
                    help="minimum number of anomalous top_features (used if --require_all_anom is not set)")
    args = ap.parse_args()

    exps = load_mapped(args.mapped)
    chosen = select_events(exps, args.require_all_anom, args.min_anom, args.max_events)
    if not chosen:
        print("[WARN] No explanations matched your filter. Try removing --require_all_anom or lowering --min_anom.")
        return

    nodes_needed = set(str(e.get("node")) for e in chosen if e.get("node") is not None)
    feats_needed = set()
    for e in chosen:
        for tf in e.get("top_features", []):
            if not tf.get("missing_in_csv", False):
                feats_needed.add(tf["feature"])

    print(f"[INFO] plotting {len(chosen)} explanations across {len(nodes_needed)} nodes; features={len(feats_needed)}")
    per_node = stream_load_nodes(args.data, nodes_needed, feats_needed, chunksize=args.chunksize)

    made = 0
    for e in chosen:
        node = str(e.get("node", ""))
        df_node = per_node.get(node)
        if df_node is None or df_node.empty:
            continue
        nid = e.get("nid", "na")
        event_date = norm_date(e.get("date"))
        out = args.out_dir / f"nid{nid}_node{node}_event{event_date}.png"
        plot_event(e, df_node, out, window=args.window, zthr=args.zthr)
        made += 1

    print(f"[OK] created {made} PNG(s) in {args.out_dir}")


if __name__ == "__main__":
    main()