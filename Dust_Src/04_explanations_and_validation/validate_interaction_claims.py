#!/usr/bin/env python3
"""
Validate interaction-mining claims with:
1) Seed-to-seed stability
2) Permutation significance
3) Temporal holdout confirmation

Expected input CSV format:
  mapped explanations flat CSV (e.g., gtg_s77_rich_mapped_explanations_flat.csv)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from mine_parameter_interactions import parse_lag_bins, mine_interactions


PairKey = Tuple[str, str, str, str]


def pair_key_from_row(r: pd.Series) -> PairKey:
    return (
        str(r["feature_a"]),
        str(r["lagbin_a"]),
        str(r["feature_b"]),
        str(r["lagbin_b"]),
    )


def pair_name(k: PairKey) -> str:
    return f"{k[0]}[{k[1]}] + {k[2]}[{k[3]}]"


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def parse_seed_input(values: List[str]) -> Dict[str, Path]:
    """
    Accepts:
      --seed-input s77=path.csv
      --seed-input path.csv
    """
    out: Dict[str, Path] = {}
    for i, v in enumerate(values):
        if "=" in v:
            k, p = v.split("=", 1)
            seed_name = k.strip()
            path = Path(p.strip())
        else:
            path = Path(v.strip())
            m = re.search(r"s(\d+)", path.name)
            seed_name = f"s{m.group(1)}" if m else f"seed{i+1}"

        if seed_name in out:
            raise ValueError(f"Duplicate seed key: {seed_name}")
        out[seed_name] = path
    return out


def mine_one_csv(
    csv_path: Path,
    topk: int,
    lag_bins: List[int],
    positive_only: bool,
    positive_threshold: float,
    drop_missing_in_csv: bool,
):
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else None
    pairs_df, ranked, lag_stats, topk_rows, summary = mine_interactions(
        df=df,
        event_col="nid",
        date_col="event_date",
        prob_col="prob",
        label_col=label_col,
        topk=topk,
        lag_bins=lag_bins,
        positive_only=positive_only,
        positive_threshold=positive_threshold,
        drop_missing_in_csv=drop_missing_in_csv,
    )
    return pairs_df, ranked, lag_stats, topk_rows, summary


def compute_seed_stability(
    seed_ranked: Dict[str, pd.DataFrame],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seeds = sorted(seed_ranked.keys())
    if len(seeds) < 2:
        return pd.DataFrame(), pd.DataFrame(), {"warning": "Need >=2 seeds for stability."}

    top_sets: Dict[str, set[PairKey]] = {}
    rank_maps: Dict[str, dict[PairKey, int]] = {}
    for s in seeds:
        r = seed_ranked[s].reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        r_top = r.head(top_n)
        keys = [pair_key_from_row(row) for _, row in r_top.iterrows()]
        top_sets[s] = set(keys)
        rank_maps[s] = {k: i + 1 for i, k in enumerate(keys)}

    rows = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            inter = len(top_sets[a].intersection(top_sets[b]))
            union = len(top_sets[a].union(top_sets[b]))
            j = inter / union if union > 0 else np.nan
            rows.append({"seed_a": a, "seed_b": b, "intersection": inter, "union": union, "jaccard_topN": j})
    pairwise = pd.DataFrame(rows).sort_values("jaccard_topN", ascending=False)

    # Consensus presence and average rank.
    all_keys = sorted(set().union(*top_sets.values()))
    stable_rows = []
    for k in all_keys:
        present = [s for s in seeds if k in top_sets[s]]
        ranks = [rank_maps[s][k] for s in present]
        stable_rows.append(
            {
                "pair": pair_name(k),
                "feature_a": k[0],
                "lagbin_a": k[1],
                "feature_b": k[2],
                "lagbin_b": k[3],
                "n_seeds_present": len(present),
                "seed_fraction": len(present) / len(seeds),
                "mean_rank_when_present": float(np.mean(ranks)),
                "max_rank_when_present": int(np.max(ranks)),
            }
        )
    stable = (
        pd.DataFrame(stable_rows)
        .sort_values(
            ["n_seeds_present", "mean_rank_when_present"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    summary = {
        "n_seeds": len(seeds),
        "top_n": int(top_n),
        "mean_pairwise_jaccard": float(pairwise["jaccard_topN"].mean()),
        "median_pairwise_jaccard": float(pairwise["jaccard_topN"].median()),
        "n_pairs_present_in_all_seeds": int((stable["n_seeds_present"] == len(seeds)).sum()),
        "n_pairs_present_in_at_least_2_seeds": int((stable["n_seeds_present"] >= 2).sum()),
    }
    return pairwise, stable, summary


def make_event_item_lists(topk_rows: pd.DataFrame) -> tuple[np.ndarray, List[np.ndarray], Dict[PairKey, int]]:
    """
    Returns:
      event_ids: ndarray of event ids (ordered groups)
      items_by_event: list of ndarray[int] containing item indices for each event
      idx_to_pair: dict index->(feature,lagbin) encoded externally
    """
    # Unique item definitions.
    item_df = (
        topk_rows[["feature", "lag_bin"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    item_df["item_idx"] = np.arange(len(item_df), dtype=int)
    topk2 = topk_rows.merge(item_df, on=["feature", "lag_bin"], how="left")

    # Per-event item lists.
    grouped = topk2.groupby("nid", sort=True)["item_idx"].apply(lambda s: s.to_numpy(dtype=int))
    event_ids = grouped.index.to_numpy()
    items_by_event = grouped.tolist()

    idx_to_pair = {
        int(r["item_idx"]): (str(r["feature"]), str(r["lag_bin"])) for _, r in item_df.iterrows()
    }
    return event_ids, items_by_event, idx_to_pair


def pair_counts_from_item_lists(
    items_by_event: List[np.ndarray],
    tracked_pairs: set[PairKey] | None = None,
    idx_to_item: Dict[int, tuple[str, str]] | None = None,
) -> Dict[PairKey, int]:
    counts: Dict[PairKey, int] = {}
    for arr in items_by_event:
        if arr.size < 2:
            continue
        # Ensure uniqueness inside event.
        uniq = np.unique(arr)
        if uniq.size < 2:
            continue

        for i in range(uniq.size):
            ai = int(uniq[i])
            for j in range(i + 1, uniq.size):
                bi = int(uniq[j])
                if idx_to_item is None:
                    raise ValueError("idx_to_item is required")
                a = idx_to_item[ai]
                b = idx_to_item[bi]
                key = tuple(sorted([a, b], key=lambda x: (x[0], x[1])))
                key4: PairKey = (key[0][0], key[0][1], key[1][0], key[1][1])
                if tracked_pairs is not None and key4 not in tracked_pairs:
                    continue
                counts[key4] = counts.get(key4, 0) + 1
    return counts


def permutation_significance(
    topk_rows: pd.DataFrame,
    ranked_pairs: pd.DataFrame,
    n_perm: int,
    top_m_pairs: int,
    rng_seed: int,
) -> pd.DataFrame:
    # Observed top pairs we test.
    tested = ranked_pairs.head(top_m_pairs).copy().reset_index(drop=True)
    tested["pair_key"] = [pair_key_from_row(r) for _, r in tested.iterrows()]
    tracked = set(tested["pair_key"].tolist())

    # Build event-item arrays.
    event_ids, items_by_event, idx_to_item = make_event_item_lists(topk_rows)
    event_sizes = np.array([len(x) for x in items_by_event], dtype=int)
    all_items = np.concatenate(items_by_event).astype(int)
    n_all = all_items.size

    # Observed counts (from ranked table directly).
    obs_count = {pair_key_from_row(r): int(r["count_events"]) for _, r in tested.iterrows()}

    rng = np.random.default_rng(rng_seed)
    perm_counts = {k: np.zeros(n_perm, dtype=int) for k in tracked}

    # Precompute split indices to preserve per-event sizes.
    split_idx = np.cumsum(event_sizes)[:-1]

    for p in range(n_perm):
        shuffled = np.array(all_items, copy=True)
        rng.shuffle(shuffled)
        chunked = np.split(shuffled, split_idx)
        cnt = pair_counts_from_item_lists(chunked, tracked_pairs=tracked, idx_to_item=idx_to_item)
        for k in tracked:
            perm_counts[k][p] = cnt.get(k, 0)

    rows = []
    for k in tested["pair_key"]:
        arr = perm_counts[k]
        obs = obs_count[k]
        p_emp = (1.0 + float(np.sum(arr >= obs))) / (n_perm + 1.0)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n_perm > 1 else 0.0
        z = (obs - mu) / sd if sd > 0 else np.nan
        rows.append(
            {
                "feature_a": k[0],
                "lagbin_a": k[1],
                "feature_b": k[2],
                "lagbin_b": k[3],
                "obs_count_events": int(obs),
                "perm_mean_count": mu,
                "perm_std_count": sd,
                "zscore_vs_perm": z,
                "p_empirical": p_emp,
            }
        )

    out = pd.DataFrame(rows).sort_values("p_empirical", ascending=True).reset_index(drop=True)
    out["q_bh_fdr"] = benjamini_hochberg(out["p_empirical"].to_numpy(dtype=float))
    out["significant_p05"] = out["p_empirical"] < 0.05
    out["significant_fdr10"] = out["q_bh_fdr"] < 0.10
    return out


def temporal_holdout_confirmation(
    source_df: pd.DataFrame,
    topk: int,
    lag_bins: List[int],
    positive_only: bool,
    positive_threshold: float,
    drop_missing_in_csv: bool,
    holdout_frac: float,
    top_n_check: int,
    min_holdout_count: int,
) -> tuple[pd.DataFrame, dict]:
    df = source_df.copy()
    if "event_date" not in df.columns:
        raise ValueError("Input CSV must contain event_date for temporal holdout.")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()

    # Split by event date quantile across unique events.
    ev_dates = (
        df[["nid", "event_date"]]
        .drop_duplicates("nid")
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    q = 1.0 - holdout_frac
    split_date = ev_dates["event_date"].quantile(q)
    ev_dates["split"] = np.where(ev_dates["event_date"] <= split_date, "discovery", "holdout")

    split_map = dict(zip(ev_dates["nid"], ev_dates["split"]))
    df["split"] = df["nid"].map(split_map)

    disc_df = df[df["split"] == "discovery"].copy()
    hold_df = df[df["split"] == "holdout"].copy()

    label_col = "label" if "label" in df.columns else None
    _, disc_ranked, _, _, disc_summary = mine_interactions(
        df=disc_df,
        event_col="nid",
        date_col="event_date",
        prob_col="prob",
        label_col=label_col,
        topk=topk,
        lag_bins=lag_bins,
        positive_only=positive_only,
        positive_threshold=positive_threshold,
        drop_missing_in_csv=drop_missing_in_csv,
    )
    _, hold_ranked, _, _, hold_summary = mine_interactions(
        df=hold_df,
        event_col="nid",
        date_col="event_date",
        prob_col="prob",
        label_col=label_col,
        topk=topk,
        lag_bins=lag_bins,
        positive_only=positive_only,
        positive_threshold=positive_threshold,
        drop_missing_in_csv=drop_missing_in_csv,
    )

    disc_top = disc_ranked.head(top_n_check).copy().reset_index(drop=True)
    if disc_top.empty:
        return pd.DataFrame(), {"warning": "Discovery set produced no ranked pairs."}

    hold_map = {
        pair_key_from_row(r): r for _, r in hold_ranked.iterrows()
    }

    rows = []
    for _, r in disc_top.iterrows():
        k = pair_key_from_row(r)
        rh = hold_map.get(k, None)
        d_count = int(r["count_events"])
        d_sup = float(r["support"])
        if rh is None:
            h_count = 0
            h_sup = 0.0
            h_lift = np.nan
        else:
            h_count = int(rh["count_events"])
            h_sup = float(rh["support"])
            h_lift = float(rh["lift"]) if "lift" in rh else np.nan

        ratio = (h_sup / d_sup) if d_sup > 0 else np.nan
        confirmed = h_count >= min_holdout_count
        rows.append(
            {
                "feature_a": k[0],
                "lagbin_a": k[1],
                "feature_b": k[2],
                "lagbin_b": k[3],
                "disc_count_events": d_count,
                "disc_support": d_sup,
                "disc_lift": float(r["lift"]) if "lift" in r else np.nan,
                "hold_count_events": h_count,
                "hold_support": h_sup,
                "hold_lift": h_lift,
                "support_ratio_hold_over_disc": ratio,
                "confirmed": bool(confirmed),
            }
        )
    out = pd.DataFrame(rows).sort_values("disc_count_events", ascending=False).reset_index(drop=True)

    summary = {
        "split_date": str(pd.Timestamp(split_date).date()),
        "n_events_discovery": int(disc_summary.get("counts", {}).get("unique_events", 0)),
        "n_events_holdout": int(hold_summary.get("counts", {}).get("unique_events", 0)),
        "top_n_checked": int(top_n_check),
        "confirmed_count": int(out["confirmed"].sum()),
        "confirmed_fraction": float(out["confirmed"].mean()),
        "min_holdout_count": int(min_holdout_count),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed-input",
        action="append",
        required=True,
        help="Repeatable. Format: seed=path.csv or just path.csv",
    )
    ap.add_argument("--reference-seed", type=str, default=None, help="Seed key used for permutation+holdout. Defaults to first key.")
    ap.add_argument("--outdir", type=Path, default=Path("interaction_validation"))
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lag-bins", type=str, default="0,7,30,90,180,365,730,1095,100000")
    ap.add_argument("--positive-only", action="store_true", default=True)
    ap.add_argument("--positive-threshold", type=float, default=0.5)
    ap.add_argument("--drop-missing-in-csv", action="store_true", default=True)
    ap.add_argument("--stability-top-n", type=int, default=50)
    ap.add_argument("--perm-n", type=int, default=500)
    ap.add_argument("--perm-top-m", type=int, default=30)
    ap.add_argument("--perm-seed", type=int, default=123)
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--holdout-top-n", type=int, default=30)
    ap.add_argument("--holdout-min-count", type=int, default=5)
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    lag_bins = parse_lag_bins(args.lag_bins)

    seed_to_path = parse_seed_input(args.seed_input)
    for s, p in seed_to_path.items():
        if not p.exists():
            raise FileNotFoundError(f"[{s}] missing file: {p}")

    # Mine each seed separately.
    seed_ranked: Dict[str, pd.DataFrame] = {}
    source_df_cache: Dict[str, pd.DataFrame] = {}
    per_seed_summary = {}
    for s, p in sorted(seed_to_path.items()):
        pairs_df, ranked, lag_stats, topk_rows, summary = mine_one_csv(
            csv_path=p,
            topk=args.topk,
            lag_bins=lag_bins,
            positive_only=args.positive_only,
            positive_threshold=args.positive_threshold,
            drop_missing_in_csv=args.drop_missing_in_csv,
        )
        seed_ranked[s] = ranked
        source_df_cache[s] = pd.read_csv(p)
        per_seed_summary[s] = summary

        ranked.to_csv(outdir / f"{s}_interaction_pairs_ranked.csv", index=False)
        pairs_df.to_csv(outdir / f"{s}_per_event_pairs.csv", index=False)
        lag_stats.to_csv(outdir / f"{s}_feature_lag_stats.csv", index=False)
        topk_rows.to_csv(outdir / f"{s}_topk_rows_per_event.csv", index=False)

    # 1) Seed stability
    pairwise, stable, stability_summary = compute_seed_stability(
        seed_ranked=seed_ranked,
        top_n=args.stability_top_n,
    )
    if not pairwise.empty:
        pairwise.to_csv(outdir / "seed_stability_pairwise_jaccard.csv", index=False)
        stable.to_csv(outdir / "seed_stable_pairs.csv", index=False)

    # Reference seed for 2+3
    ref_seed = args.reference_seed or sorted(seed_to_path.keys())[0]
    if ref_seed not in seed_to_path:
        raise ValueError(f"reference seed '{ref_seed}' not in provided --seed-input keys")
    ref_ranked = seed_ranked[ref_seed]

    # Need topk rows from reference in exact mined configuration.
    _, _, _, ref_topk_rows, ref_summary = mine_one_csv(
        csv_path=seed_to_path[ref_seed],
        topk=args.topk,
        lag_bins=lag_bins,
        positive_only=args.positive_only,
        positive_threshold=args.positive_threshold,
        drop_missing_in_csv=args.drop_missing_in_csv,
    )

    # 2) Permutation significance
    perm_df = permutation_significance(
        topk_rows=ref_topk_rows,
        ranked_pairs=ref_ranked,
        n_perm=args.perm_n,
        top_m_pairs=args.perm_top_m,
        rng_seed=args.perm_seed,
    )
    perm_df.to_csv(outdir / "permutation_significance.csv", index=False)
    perm_summary = {
        "reference_seed": ref_seed,
        "tested_top_m_pairs": int(min(args.perm_top_m, len(ref_ranked))),
        "n_permutations": int(args.perm_n),
        "n_significant_p05": int(perm_df["significant_p05"].sum()),
        "n_significant_fdr10": int(perm_df["significant_fdr10"].sum()),
    }

    # 3) Temporal holdout confirmation (on reference seed)
    ref_source_df = source_df_cache[ref_seed]
    hold_df, hold_summary = temporal_holdout_confirmation(
        source_df=ref_source_df,
        topk=args.topk,
        lag_bins=lag_bins,
        positive_only=args.positive_only,
        positive_threshold=args.positive_threshold,
        drop_missing_in_csv=args.drop_missing_in_csv,
        holdout_frac=args.holdout_frac,
        top_n_check=args.holdout_top_n,
        min_holdout_count=args.holdout_min_count,
    )
    if not hold_df.empty:
        hold_df.to_csv(outdir / "temporal_holdout_confirmation.csv", index=False)

    final_summary = {
        "inputs": {
            "seed_inputs": {k: str(v) for k, v in seed_to_path.items()},
            "reference_seed": ref_seed,
            "topk": int(args.topk),
            "lag_bins": lag_bins,
            "positive_only": bool(args.positive_only),
            "positive_threshold": float(args.positive_threshold),
            "drop_missing_in_csv": bool(args.drop_missing_in_csv),
            "stability_top_n": int(args.stability_top_n),
            "perm_n": int(args.perm_n),
            "perm_top_m": int(args.perm_top_m),
            "holdout_frac": float(args.holdout_frac),
            "holdout_top_n": int(args.holdout_top_n),
            "holdout_min_count": int(args.holdout_min_count),
        },
        "per_seed_summary": per_seed_summary,
        "stability_summary": stability_summary,
        "permutation_summary": perm_summary,
        "holdout_summary": hold_summary,
    }

    with open(outdir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print(f"[OK] wrote validation outputs to: {outdir}")
    if stability_summary.get("mean_pairwise_jaccard") is not None:
        print(
            f"[stability] mean pairwise Jaccard@{args.stability_top_n}: "
            f"{stability_summary['mean_pairwise_jaccard']:.3f}"
        )
    print(
        f"[permutation] significant pairs: p<0.05={perm_summary['n_significant_p05']}, "
        f"FDR<0.10={perm_summary['n_significant_fdr10']}"
    )
    if hold_summary:
        print(
            f"[holdout] confirmed top-{args.holdout_top_n} pairs: "
            f"{hold_summary.get('confirmed_count', 0)}/{hold_summary.get('top_n_checked', 0)}"
        )


if __name__ == "__main__":
    main()

