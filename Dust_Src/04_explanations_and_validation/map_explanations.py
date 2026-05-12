#!/usr/bin/env python3
"""map_explanations.py

Map explanation outputs to (node -> lon/lat) and evaluate whether each feature value
at its lag day was anomalous for that node (node-wise z-score).

Assumptions (matches your dataset builder):
- nodes are "i_j" where i in [0..19], j in [0..19]
- local_node is the flattened index in row-major order: idx = j*NX + i (0-based)

Usage:
  python3 map_explanations.py \
    --data GRID3km_DAILY_20x20_modelready_labeled_featured.csv \
    --explanations explanations.json \
    --out_json mapped_explanations.json \
    --out_csv  mapped_explanations_flat.csv \
    --zthr 2.0

Notes:
- Streams the dataset to avoid loading the full CSV into RAM.
- Computes per-node mean/std for ONLY the features that appear in explanations.
- Extracts the exact (node, lag_date) feature values needed for the explanation rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd

# ---- grid constants (must match your dataset code) ----
REGION = (45.389892, 31.230722, 45.928881, 31.690477)  # (lon_min, lat_min, lon_max, lat_max)
NX = NY = 20


def build_node_geo_df(region=REGION, nx=NX, ny=NY) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = region
    dx = (lon_max - lon_min) / nx
    dy = (lat_max - lat_min) / ny
    rows = []
    idx = 0
    for j in range(ny):
        lat = lat_min + (j + 0.5) * dy
        for i in range(nx):
            lon = lon_min + (i + 0.5) * dx
            rows.append({
                "local_node": idx,
                "i": i,
                "j": j,
                "node": f"{i}_{j}",
                "longitude": lon,
                "latitude": lat,
            })
            idx += 1
    return pd.DataFrame(rows)


def node_from_local(local_node: int, nx=NX, ny=NY, one_based: bool = False) -> str:
    if one_based:
        local_node -= 1
    if local_node < 0 or local_node >= nx * ny:
        raise ValueError(f"local_node={local_node} out of range for {nx}x{ny}")
    i = int(local_node % nx)
    j = int(local_node // nx)
    return f"{i}_{j}"


def normalize_date_str(s: Any) -> str:
    # keep 'YYYY-MM-DD' regardless of presence of time
    if s is None:
        return ""
    return str(s)[:10]


def load_explanations(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        exps = json.load(f)
    if not isinstance(exps, list):
        raise ValueError("explanations JSON must be a list of objects")
    return exps


def collect_requirements(
    exps: List[Dict[str, Any]],
    one_based_local_node: bool = False
) -> Tuple[Set[str], Set[str], Set[str]]:
    nodes_needed: Set[str] = set()
    dates_needed: Set[str] = set()
    feats_needed: Set[str] = set()

    for e in exps:
        ln = int(e["local_node"])
        node = node_from_local(ln, one_based=one_based_local_node)
        nodes_needed.add(node)

        if "date" in e:
            dates_needed.add(normalize_date_str(e["date"]))

        for tf in e.get("top_features", []):
            feats_needed.add(tf["feature"])
            dates_needed.add(normalize_date_str(tf.get("lag_date")))
    return nodes_needed, dates_needed, feats_needed


def stream_stats_and_extract_values(
    data_csv: Path,
    nodes_needed: Set[str],
    dates_needed: Set[str],
    feats_needed: Set[str],
    chunksize: int = 500_000,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]], Set[str]]:
    """
    One streaming pass:
      - compute per-node mean/std for feats_needed
      - extract rows for (node in nodes_needed AND date in dates_needed)
    Returns:
      extracted_df: columns = ['node','date'] + feats_available (only rows we need)
      stats: dict feature -> (mean[400], std[400]) arrays (only for feats_available)
      feats_missing: features referenced in explanations but not present in CSV
    """

    header_cols = pd.read_csv(data_csv, nrows=0).columns.tolist()
    header_set = set(header_cols)
    orig_feats = set(feats_needed)

    feats_available = sorted([f for f in feats_needed if f in header_set])
    feats_missing = set([f for f in orig_feats if f not in header_set])

    if feats_missing:
        ex = sorted(list(feats_missing))[:10]
        print(f"[WARN] {len(feats_missing)} features referenced in explanations are missing from CSV. Example: {ex}")

    usecols = ["node", "date"] + feats_available

    # map node -> idx [0..399]
    geo = build_node_geo_df()
    node_to_idx = dict(zip(geo["node"].tolist(), geo["local_node"].tolist()))
    n_nodes = NX * NY

    # accumulators for mean/std per node+feature
    counts = {f: np.zeros(n_nodes, dtype=np.int64) for f in feats_available}
    sums = {f: np.zeros(n_nodes, dtype=np.float64) for f in feats_available}
    sumsqs = {f: np.zeros(n_nodes, dtype=np.float64) for f in feats_available}

    extracted_parts = []

    for chunk in pd.read_csv(data_csv, usecols=usecols, chunksize=chunksize):
        # normalize date to YYYY-MM-DD string
        chunk["date"] = chunk["date"].astype(str).str.slice(0, 10)
        chunk["node"] = chunk["node"].astype(str)

        idx = chunk["node"].map(node_to_idx).to_numpy()
        valid_node = ~pd.isna(idx)
        idx = idx[valid_node].astype(np.int64)
        if idx.size == 0:
            continue

        for f in feats_available:
            v = chunk.loc[valid_node, f].to_numpy(dtype=np.float64, copy=False)
            m = ~np.isnan(v)
            if not m.any():
                continue
            idxm = idx[m]
            vm = v[m]
            counts[f] += np.bincount(idxm, minlength=n_nodes)
            sums[f] += np.bincount(idxm, weights=vm, minlength=n_nodes)
            sumsqs[f] += np.bincount(idxm, weights=vm * vm, minlength=n_nodes)

        mask_extract = chunk["node"].isin(nodes_needed) & chunk["date"].isin(dates_needed)
        if mask_extract.any():
            extracted_parts.append(chunk.loc[mask_extract, ["node", "date"] + feats_available])

    extracted_df = (
        pd.concat(extracted_parts, ignore_index=True)
        if extracted_parts
        else pd.DataFrame(columns=["node", "date"] + feats_available)
    )
    if len(extracted_df):
        extracted_df = (
            extracted_df.sort_values(["date", "node"])
            .drop_duplicates(subset=["node", "date"], keep="last")
            .reset_index(drop=True)
        )

    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in feats_available:
        c = counts[f].astype(np.float64)
        mean = np.divide(sums[f], c, out=np.full_like(sums[f], np.nan), where=c > 0)
        ex2 = np.divide(sumsqs[f], c, out=np.full_like(sumsqs[f], np.nan), where=c > 0)
        var = ex2 - mean * mean
        var = np.where(var < 0, 0, var)  # numeric safety
        std = np.sqrt(var)
        stats[f] = (mean, std)

    return extracted_df, stats, feats_missing


def enrich_explanations(
    exps: List[Dict[str, Any]],
    extracted_df: pd.DataFrame,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feats_missing: Set[str],
    zthr: float = 2.0,
    one_based_local_node: bool = False
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Adds: node string, lon/lat, and for each top_feature adds
      - data_value (from CSV at lag_date)
      - node_mean/node_std (for that feature)
      - zscore
      - is_anomalous (abs(zscore)>=zthr)
      - value_mismatch (data_value - explanation_value) if both exist
      - missing_in_csv flag (if feature not in CSV)
    Also returns a flattened table for analysis.
    """
    geo = build_node_geo_df()
    geo_idx = geo.set_index("local_node")

    if len(extracted_df):
        extracted_map = extracted_df.set_index(["node", "date"])
    else:
        extracted_map = pd.DataFrame().set_index(
            pd.MultiIndex.from_arrays([[], []], names=["node", "date"])
        )

    flat_rows = []
    out = []

    for e in exps:
        ln = int(e["local_node"])
        node = node_from_local(ln, one_based=one_based_local_node)
        idx = int(ln if not one_based_local_node else ln - 1)

        g = geo_idx.loc[idx]
        e2 = dict(e)
        e2["node"] = node
        e2["longitude"] = float(g["longitude"])
        e2["latitude"] = float(g["latitude"])

        new_tfs = []
        for tf in e.get("top_features", []):
            f = tf["feature"]
            lag_date = normalize_date_str(tf.get("lag_date"))

            missing_in_csv = f in feats_missing or f not in stats

            data_value = np.nan
            if not missing_in_csv and (node, lag_date) in extracted_map.index and f in extracted_map.columns:
                dv = extracted_map.loc[(node, lag_date), f]
                try:
                    if hasattr(dv, "__len__") and not np.isscalar(dv):
                        dv = np.array(dv)[-1]
                    data_value = float(dv)
                except Exception:
                    pass

            mu = sd = z = np.nan
            is_anom = False
            if not missing_in_csv:
                mean_arr, std_arr = stats[f]
                mu = float(mean_arr[idx]) if not np.isnan(mean_arr[idx]) else np.nan
                sd = float(std_arr[idx]) if not np.isnan(std_arr[idx]) else np.nan
                if not np.isnan(data_value) and not np.isnan(mu) and sd > 0:
                    z = float((data_value - mu) / sd)
                    is_anom = bool(abs(z) >= zthr)

            value_mismatch = np.nan
            try:
                ev = float(tf.get("value"))
                if not np.isnan(data_value):
                    value_mismatch = float(data_value - ev)
            except Exception:
                pass

            tf2 = dict(tf)
            tf2.update({
                "node": node,
                "lag_date": lag_date,
                "missing_in_csv": bool(missing_in_csv),
                "data_value": (None if np.isnan(data_value) else float(data_value)),
                "node_mean": (None if np.isnan(mu) else float(mu)),
                "node_std": (None if np.isnan(sd) else float(sd)),
                "zscore": (None if np.isnan(z) else float(z)),
                "is_anomalous": is_anom,
                "value_mismatch": (None if np.isnan(value_mismatch) else float(value_mismatch)),
            })
            new_tfs.append(tf2)

            flat_rows.append({
                "nid": e.get("nid"),
                "local_node": ln,
                "node": node,
                "longitude": float(g["longitude"]),
                "latitude": float(g["latitude"]),
                "event_date": normalize_date_str(e.get("date", "")),
                "time_index": e.get("time_index"),
                "label": e.get("label"),
                "prob": e.get("prob"),
                "feature": f,
                "group": tf.get("group"),
                "lag_days": tf.get("lag_days"),
                "lag_date": lag_date,
                "missing_in_csv": bool(missing_in_csv),
                "value_expl": tf.get("value"),
                "value_data": (None if np.isnan(data_value) else float(data_value)),
                "node_mean": (None if np.isnan(mu) else float(mu)),
                "node_std": (None if np.isnan(sd) else float(sd)),
                "zscore": (None if np.isnan(z) else float(z)),
                "is_anomalous": is_anom,
                "attention": tf.get("attention"),
                "contribution": tf.get("contribution"),
                "value_mismatch": (None if np.isnan(value_mismatch) else float(value_mismatch)),
            })

        e2["top_features"] = new_tfs
        out.append(e2)

    flat_df = pd.DataFrame(flat_rows)
    return out, flat_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="modelready CSV path")
    ap.add_argument("--explanations", required=True, type=Path, help="explanations JSON path (list)")
    ap.add_argument("--out_json", required=True, type=Path, help="output JSON path")
    ap.add_argument("--out_csv", default=None, type=Path, help="optional flattened CSV output")
    ap.add_argument("--chunksize", default=500_000, type=int)
    ap.add_argument("--zthr", default=2.0, type=float, help="abs(zscore) threshold")
    ap.add_argument("--one_based_local_node", action="store_true",
                    help="set if your local_node starts at 1 (rare).")
    args = ap.parse_args()

    exps = load_explanations(args.explanations)
    nodes_needed, dates_needed, feats_needed = collect_requirements(
        exps, one_based_local_node=args.one_based_local_node
    )

    extracted_df, stats, feats_missing = stream_stats_and_extract_values(
        data_csv=args.data,
        nodes_needed=nodes_needed,
        dates_needed=dates_needed,
        feats_needed=feats_needed,
        chunksize=args.chunksize,
    )

    enriched, flat_df = enrich_explanations(
        exps=exps,
        extracted_df=extracted_df,
        stats=stats,
        feats_missing=feats_missing,
        zthr=args.zthr,
        one_based_local_node=args.one_based_local_node,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    if args.out_csv:
        flat_df.to_csv(args.out_csv, index=False)

    print(f"[OK] wrote {args.out_json}")
    if args.out_csv:
        print(f"[OK] wrote {args.out_csv}")


if __name__ == "__main__":
    main()