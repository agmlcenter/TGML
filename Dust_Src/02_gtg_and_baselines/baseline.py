#!/usr/bin/env python3
"""
Simple non-graph baselines for daily dust activation:

Baselines:
  1) global_mean : constant predictor = global event rate on train
  2) node_mean   : per-node historical mean label (train only)
  3) xgboost     : gradient boosting on per-day, per-node features

Splits:
  - Train / Val / Test by time index over dates: 70% / 15% / 15%
Metrics:
  - ROC-AUC, PR-AUC, F1 (at best threshold on val), ACC, PREC, REC
  - High-wind subsets on test: p90, p95, p98, p99 of wind_feature

Input CSV must contain:
  - 'node'  : node id (e.g. '7_4')
  - 'date'  : date string (parsable by pandas)
  - 'label' : binary or {0,1}-like target
  - other numeric feature columns (including wind_feature)
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

import xgboost as xgb


# ---------------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------------

def load_daily_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"node", "date", "label"}
    assert required.issubset(df.columns), f"CSV must have columns {required}"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "node"]).reset_index(drop=True)
    return df


def normalize_features(X: np.ndarray):
    """
    Standardize features to mean 0, std 1 (per column), robust to NaNs/Infs.
    NaNs/Infs are converted to finite values at the end.
    """
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std = std + 1e-8
    Xn = (X - mean) / std
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=10.0, neginf=-10.0)
    return Xn, mean, std


def make_time_splits(df: pd.DataFrame):
    """
    Time-based splits by 'date': 70% / 15% / 15% of unique dates.
    Returns boolean masks (numpy arrays) aligned with df rows.
    """
    unique_dates = np.sort(df["date"].unique())
    T = len(unique_dates)
    train_T = int(0.7 * T)
    val_T = int(0.85 * T)

    train_dates = set(unique_dates[:train_T])
    val_dates = set(unique_dates[train_T:val_T])
    test_dates = set(unique_dates[val_T:])

    dates = df["date"].values
    train_mask = np.isin(dates, list(train_dates))
    val_mask = np.isin(dates, list(val_dates))
    test_mask = np.isin(dates, list(test_dates))

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Metrics helpers (same spirit as graph script)
# ---------------------------------------------------------------------------

def best_f1_threshold(y_true, prob):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    mask = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[mask]
    prob = prob[mask]

    if y_true.size == 0:
        return 0.5, float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.nanargmax(f1)
    if idx >= len(thresholds):
        thr = 0.5
    else:
        thr = thresholds[idx]
    return float(thr), float(f1[idx])


def compute_basic_metrics(y_true, prob, thr=0.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    mask = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[mask]
    prob = prob[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "pos_rate": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "f1": np.nan,
            "acc": np.nan,
            "prec": np.nan,
            "rec": np.nan,
            "thr_f1": thr,
        }

    try:
        roc = roc_auc_score(y_true, prob)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, prob)
    except ValueError:
        pr = float("nan")

    y_pred = (prob >= thr).astype(int)
    f1 = f1_score(y_true, y_pred)
    acc = (y_true == y_pred).mean()

    if y_pred.sum() > 0:
        prec = (y_true[y_pred == 1].sum()) / y_pred.sum()
    else:
        prec = 0.0
    if y_true.sum() > 0:
        rec = (y_pred[y_true == 1].sum()) / y_true.sum()
    else:
        rec = 0.0

    return {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "f1": float(f1),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "thr_f1": float(thr),
    }


def evaluate_probs(y, prob, wind, train_mask, val_mask, test_mask, name="baseline"):
    """
    Unified evaluation: splits + high-wind subsets.
    """
    y = np.nan_to_num(y.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.nan_to_num(prob.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    wind = np.nan_to_num(wind.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }

    # Best threshold on val
    if val_mask.sum() > 0:
        thr_best, f1_val = best_f1_threshold(y[val_mask], prob[val_mask])
    else:
        thr_best, f1_val = (0.5, float("nan"))

    results = {
        "best_thr": thr_best,
        "val_f1_at_best_thr": f1_val,
        "splits": {},
        "highwind": {},
    }

    print("=" * 80)
    print(f"Baseline: {name}")
    print("=" * 80)
    print(f"  best_thr (from val) = {thr_best:.4f}, val_f1 = {f1_val:.4f}")

    # Global splits
    for split_name, m in masks.items():
        res_split = compute_basic_metrics(y[m], prob[m], thr=thr_best)
        results["splits"][split_name] = res_split
        print(
            f"[{split_name}] n={res_split['n']} pos_rate={res_split['pos_rate']:.6f} "
            f"ROC-AUC={res_split['roc_auc']:.4f} PR-AUC={res_split['pr_auc']:.4f} "
            f"F1@thr={res_split['f1']:.4f}"
        )

    # High-wind subsets on test
    m_test = test_mask
    y_test = y[m_test]
    p_test = prob[m_test]
    w_test = wind[m_test]

    if len(y_test) > 0 and np.any(w_test != 0.0):
        for p in [90, 95, 98, 99]:
            thr_w = np.percentile(w_test, p)
            hw_mask = (w_test >= thr_w)
            y_hw = y_test[hw_mask]
            p_hw = p_test[hw_mask]
            res_hw = compute_basic_metrics(y_hw, p_hw, thr=thr_best)
            key = f"p{p}"
            results["highwind"][key] = res_hw
            print(
                f"[test highwind {key}] n={res_hw['n']} pos_rate={res_hw['pos_rate']:.6f} "
                f"ROC-AUC={res_hw['roc_auc']:.4f} PR-AUC={res_hw['pr_auc']:.4f} "
                f"F1@thr={res_hw['f1']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Baseline builders
# ---------------------------------------------------------------------------

def run_global_mean_baseline(df, y, wind, train_mask, val_mask, test_mask):
    # Global mean from train
    y_train = y[train_mask]
    y_train_clean = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
    global_mean = float(y_train_clean.mean())
    prob = np.full_like(y, global_mean, dtype=float)
    return evaluate_probs(y, prob, wind, train_mask, val_mask, test_mask, name="global_mean")


def run_node_mean_baseline(df, y, wind, train_mask, val_mask, test_mask):
    df_tmp = df.copy()
    df_tmp["label_clean"] = np.nan_to_num(df_tmp["label"].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)

    df_train = df_tmp[train_mask]
    node_mean = df_train.groupby("node")["label_clean"].mean()
    global_mean = float(df_train["label_clean"].mean())

    prob = df_tmp["node"].map(node_mean).fillna(global_mean).values.astype(float)
    return evaluate_probs(y, prob, wind, train_mask, val_mask, test_mask, name="node_mean")


def run_xgboost_baseline(df, y, wind, train_mask, val_mask, test_mask):
    # Numeric feature columns excluding node/date/label
    exclude_cols = {"node", "date", "label"}
    feat_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    print(f"XGBoost baseline using {len(feat_cols)} numeric features.")

    X = df[feat_cols].values.astype(np.float32)
    X_norm, _, _ = normalize_features(X)

    y_clean = np.nan_to_num(y.astype(float), nan=0.0, posinf=1.0, neginf=0.0)

    # Class imbalance
    y_train = y_clean[train_mask]
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    if pos > 0:
        scale_pos_weight = neg / (pos + 1e-8)
    else:
        scale_pos_weight = 1.0
    print(f"XGBoost: train pos_rate={pos / (pos + neg + 1e-8):.6f}, scale_pos_weight~{scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",      # change to 'gpu_hist' if you want GPU
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=8,
	verbose=True,
        early_stopping_rounds=50,
    )

    X_train = X_norm[train_mask]
    y_train = y_clean[train_mask]
    X_val = X_norm[val_mask]
    y_val = y_clean[val_mask]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)]
    )

    prob = model.predict_proba(X_norm)[:, 1]
    return evaluate_probs(y_clean, prob, wind, train_mask, val_mask, test_mask, name="xgboost")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to daily CSV")
    parser.add_argument("--wind-feature", type=str, default="wind_speed", help="Column name for wind magnitude")
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = load_daily_csv(args.csv)

    train_mask, val_mask, test_mask = make_time_splits(df)
    print(f"Split sizes: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # Targets
    y = df["label"].values.astype(float)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

    # Wind values for high-wind subsets
    if args.wind_feature in df.columns:
        wind = df[args.wind_feature].values.astype(float)
    else:
        print(f"WARNING: wind feature '{args.wind_feature}' not in CSV; high-wind metrics will use zeros.")
        wind = np.zeros(len(df), dtype=float)

    all_results = {}

    # 1) Global mean baseline
    res_global = run_global_mean_baseline(df, y, wind, train_mask, val_mask, test_mask)
    all_results["global_mean"] = res_global

    # 2) Node-mean baseline
    res_node = run_node_mean_baseline(df, y, wind, train_mask, val_mask, test_mask)
    all_results["node_mean"] = res_node

    # 3) XGBoost baseline
    res_xgb = run_xgboost_baseline(df, y, wind, train_mask, val_mask, test_mask)
    all_results["xgboost"] = res_xgb

    # Global summary (test split)
    print("\n=== Global summary (test split) ===")
    for key, res in all_results.items():
        stats = res["splits"]["test"]
        print(
            f"{key:15s} | ROC-AUC={stats['roc_auc']:.4f} "
            f"PR-AUC={stats['pr_auc']:.4f} F1={stats['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
