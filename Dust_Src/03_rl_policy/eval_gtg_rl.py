#!/usr/bin/env python3
"""
Evaluate PPO RL classifier on GTG outputs, using the SAME setup as train_gtg_rl.py.

- Loads GTG RL data .npz produced by gtg.py
- Rebuilds the RL region with --gtg-train-years and 70/15/15 split
- Loads PPO model
- Computes p_alert and applies threshold (from --thr or chosen_threshold.json)
- Reports node-level metrics + 5-day early-warning metrics
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)


# ---------------------------------------------------------------------
# Early-warning metrics (copied from gtg.py to be self-contained)
# ---------------------------------------------------------------------

@dataclass
class EarlyWarningStats:
    events: int
    hit_rate: float
    miss_rate: float
    mean_lead: float
    median_lead: float
    window_days: int
    thr: float
    fa_rate: float        # fraction of negative nodes that fire
    fa_count: int         # number of false alarms (negative nodes firing)
    fa_per_event: float   # fa_count / events (how many false alarms per event)


@torch.no_grad()
def policy_p_alert(model: PPO, obs_np: np.ndarray) -> np.ndarray:
    """Return P(action=alert) from PPO policy for each observation."""
    obs_t = obs_as_tensor(obs_np, model.device)
    dist = model.policy.get_distribution(obs_t)
    probs = dist.distribution.probs  # shape [B, 2] for Discrete(2)
    return probs[:, 1].detach().cpu().numpy()


def compute_early_warning_stats(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_idx: np.ndarray,
    thr: float,
    window_days: int = 5,
) -> EarlyWarningStats:
    """
    5-day early-warning metrics + false-alarm stats.

    Event definition (node-level):
      - Each index i with y_true[i] = 1 is an event at time t_event = time_idx[i].
      - For that event, look back 'window_days' (inclusive):
          [t_event - window_days, t_event]
        If there is ANY node j with y_score[j] >= thr in that window, the event
        is a 'hit'. Lead time:
          lead = t_event - t_first_alert
        where t_first_alert is the earliest alert in that window.

    False-alarm stats:
      - Negatives: y_true == 0.
      - fa_count = #negatives with y_score >= thr.
      - fa_rate  = fa_count / #negatives.
      - fa_per_event = fa_count / events  (if events > 0).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    time_idx = np.asarray(time_idx)

    event_idx = np.where(y_true == 1)[0]
    n_events = int(event_idx.size)

    alerts = (y_score >= thr)
    lead_times = []
    hits = 0

    # Edge case: no events
    if n_events == 0:
        neg_mask = (y_true == 0)
        n_neg = int(neg_mask.sum())
        if n_neg > 0:
            fa_count = int(np.logical_and(alerts, neg_mask).sum())
            fa_rate = fa_count / n_neg
        else:
            fa_count = 0
            fa_rate = float("nan")

        return EarlyWarningStats(
            events=0,
            hit_rate=float("nan"),
            miss_rate=float("nan"),
            mean_lead=float("nan"),
            median_lead=float("nan"),
            window_days=window_days,
            thr=thr,
            fa_rate=fa_rate,
            fa_count=fa_count,
            fa_per_event=float("nan"),
        )

    # Loop over all events
    for e in event_idx:
        t_event = time_idx[e]
        t_min = t_event - window_days

        in_window = (time_idx >= t_min) & (time_idx <= t_event)
        window_alerts = alerts & in_window

        if np.any(window_alerts):
            hits += 1
            alert_times = time_idx[window_alerts]
            t_first = alert_times.min()
            lead_times.append(float(t_event - t_first))

    hit_rate = hits / n_events
    miss_rate = 1.0 - hit_rate

    if lead_times:
        mean_lead = float(np.mean(lead_times))
        median_lead = float(np.median(lead_times))
    else:
        mean_lead = 0.0
        median_lead = 0.0

    neg_mask = (y_true == 0)
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        fa_count = int(np.logical_and(alerts, neg_mask).sum())
        fa_rate = fa_count / n_neg
    else:
        fa_count = 0
        fa_rate = 0.0

    fa_per_event = fa_count / n_events if n_events > 0 else float("nan")

    return EarlyWarningStats(
        events=n_events,
        hit_rate=hit_rate,
        miss_rate=miss_rate,
        mean_lead=mean_lead,
        median_lead=median_lead,
        window_days=window_days,
        thr=thr,
        fa_rate=fa_rate,
        fa_count=fa_count,
        fa_per_event=fa_per_event,
    )


# ---------------------------------------------------------------------
# Helper: build RL region and masks exactly as in train script
# ---------------------------------------------------------------------

def build_rl_region(raw, total_years: float = 20.0, gtg_train_years: float = 12.0):
    """
    Match train_gtg_rl.py exactly:
      - RL region = suffix where time_idx >= T_cut
      - T_cut = int(T_total * gtg_train_years / total_years)
      - RL split inside region = 70/15/15 by time
    """
    prob = raw["prob"].astype(np.float32)
    y = raw["y"].astype(int)
    time_idx = raw["time_idx"].astype(int)
    x = raw["x"].astype(np.float32)

    T_total = int(time_idx.max() + 1)
    T_cut = int(T_total * gtg_train_years / total_years)

    rl_mask = time_idx >= T_cut
    x_rl = x[rl_mask]
    prob_rl = prob[rl_mask]
    y_rl = y[rl_mask]
    time_rl = time_idx[rl_mask]

    t0 = int(time_rl.min())
    Trl = int(time_rl.max() - t0 + 1)
    train_T = t0 + int(0.70 * Trl)
    val_T = t0 + int(0.85 * Trl)

    mask_train_rl = time_rl < train_T
    mask_val_rl = (time_rl >= train_T) & (time_rl < val_T)
    mask_test_rl = time_rl >= val_T

    meta = {
        "T_total": T_total,
        "T_cut": T_cut,
        "Trl": Trl,
        "train_T": train_T,
        "val_T": val_T,
    }
    return x_rl, prob_rl, y_rl, time_rl, mask_train_rl, mask_val_rl, mask_test_rl, meta


def resolve_threshold(model_path: str, thr_arg: float | None, threshold_json: str | None):
    if thr_arg is not None:
        return float(thr_arg), "cli(--thr)"

    candidates = []
    if threshold_json:
        candidates.append(threshold_json)
    candidates.append(os.path.join(os.path.dirname(model_path), "chosen_threshold.json"))

    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "thr" in obj:
                return float(obj["thr"]), f"json({p})"

    return 0.5, "default(0.5)"


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_rl_model(
    npz_path: str,
    model_path: str,
    total_years: float = 20.0,
    gtg_train_years: float = 12.0,
    ew_window_days: int = 5,
    thr: float | None = None,
    threshold_json: str | None = None,
    device: str = "cpu",
    show_action_metrics: bool = False,
):
    raw = np.load(npz_path, allow_pickle=True)

    (
        x_rl,
        prob_rl,
        y_rl,
        time_rl,
        mask_train_rl,
        mask_val_rl,
        mask_test_rl,
        meta,
    ) = build_rl_region(raw, total_years=total_years, gtg_train_years=gtg_train_years)

    obs_all = np.concatenate(
        [prob_rl.reshape(-1, 1), x_rl], axis=1
    ).astype(np.float32)

    print(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path, device=device)

    idx_test = np.where(mask_test_rl)[0]
    y_true = y_rl[idx_test].astype(int)
    obs_test = obs_all[idx_test]
    time_test = time_rl[idx_test]

    print(
        f"Split info: T_total={meta['T_total']} T_cut={meta['T_cut']} "
        f"RL_T={meta['Trl']} (train< {meta['train_T']}, val< {meta['val_T']}, test>= {meta['val_T']})"
    )
    print(
        f"RL counts: train={int(mask_train_rl.sum())} "
        f"val={int(mask_val_rl.sum())} test={int(mask_test_rl.sum())}"
    )
    print(f"RL test size: {len(idx_test)} nodes")

    score_test = policy_p_alert(model, obs_test)
    thr_eff, thr_src = resolve_threshold(model_path=model_path, thr_arg=thr, threshold_json=threshold_json)
    y_pred = (score_test >= thr_eff).astype(int)
    print(f"Using threshold: thr={thr_eff:.4f} from {thr_src}")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== RL PPO Evaluation on RL TEST subset (node-level) ===")
    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1-score:  {f1:.6f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    ew = compute_early_warning_stats(
        y_true=y_true,
        y_score=score_test,
        time_idx=time_test,
        thr=thr_eff,
        window_days=ew_window_days,
    )

    print(f"\n=== RL PPO Early-warning metrics ({ew_window_days}-day window, thr={thr_eff:.4f}) ===")
    print(f"Events (y=1)        : {ew.events}")
    print(f"Hit rate            : {ew.hit_rate:.3f}")
    print(f"Miss rate           : {ew.miss_rate:.3f}")
    print(f"Mean lead (days)    : {ew.mean_lead:.2f}")
    print(f"Median lead (days)  : {ew.median_lead:.2f}")
    print(f"False-alarm rate    : {ew.fa_rate:.5f}")
    print(f"False alarms total  : {ew.fa_count}")
    print(f"False alarms / event: {ew.fa_per_event:.2f}")

    if show_action_metrics:
        actions, _ = model.predict(obs_test, deterministic=True)
        y_act = np.asarray(actions).reshape(-1).astype(int)

        acc_a = accuracy_score(y_true, y_act)
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
            y_true, y_act, average="binary", zero_division=0
        )
        cm_a = confusion_matrix(y_true, y_act)

        print("\n=== Reference: deterministic action metrics (argmax policy) ===")
        print(f"Accuracy:  {acc_a:.6f}")
        print(f"Precision: {prec_a:.6f}")
        print(f"Recall:    {rec_a:.6f}")
        print(f"F1-score:  {f1_a:.6f}")
        print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
        print(cm_a)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        default="gtg_tgcn_rl_data.npz",
        help="Path to GTG-TGCN RL data npz file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_gtg_rl_final.zip",
        help="Path to trained PPO model .zip",
    )
    parser.add_argument(
        "--gtg-train-years",
        type=float,
        default=12.0,
        help="Must match train_gtg_rl.py: years used by GTG before RL suffix starts",
    )
    parser.add_argument(
        "--rl-years",
        type=float,
        default=None,
        help="Deprecated compatibility arg. If set, gtg_train_years = total_years - rl_years",
    )
    parser.add_argument(
        "--total-years",
        type=float,
        default=20.0,
        help="Total number of years in the dataset",
    )
    parser.add_argument(
        "--ew-window-days",
        type=int,
        default=5,
        help="Early-warning window in days (must match gtg.py for fair comparison)",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Optional manual threshold on p_alert. If omitted, tries chosen_threshold.json.",
    )
    parser.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help="Optional path to JSON containing {'thr': ...}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PPO load/inference: cpu or cuda",
    )
    parser.add_argument(
        "--show-action-metrics",
        action="store_true",
        help="Also report deterministic action(argmax) metrics for debugging.",
    )
    args = parser.parse_args()

    gtg_train_years = args.gtg_train_years
    if args.rl_years is not None:
        gtg_train_years = float(args.total_years) - float(args.rl_years)

    evaluate_rl_model(
        npz_path=args.npz,
        model_path=args.model,
        total_years=args.total_years,
        gtg_train_years=gtg_train_years,
        ew_window_days=args.ew_window_days,
        thr=args.thr,
        threshold_json=args.threshold_json,
        device=args.device,
        show_action_metrics=args.show_action_metrics,
    )


if __name__ == "__main__":
    main()
