#!/usr/bin/env python3
import argparse
import json

import numpy as np
import torch
from stable_baselines3 import PPO


def build_rl_region(time_idx: np.ndarray,
                    total_years: float,
                    gtg_train_years: float):
    """
    Reproduce the RL region and (train/val/test) time splits
    exactly as in train_gtg_rl.py.
    """
    T_total = int(time_idx.max() + 1)
    T_cut = int(T_total * gtg_train_years / total_years)

    rl_mask = time_idx >= T_cut
    time_rl = time_idx[rl_mask]

    t0 = int(time_rl.min())
    Trl = int(time_rl.max() - t0 + 1)

    train_T = t0 + int(0.70 * Trl)
    val_T = t0 + int(0.85 * Trl)

    rl_train_mask = time_rl < train_T
    rl_val_mask = (time_rl >= train_T) & (time_rl < val_T)
    rl_test_mask = time_rl >= val_T

    return rl_mask, rl_train_mask, rl_val_mask, rl_test_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str,
                        default="gtg_tgcn_rl_data.npz",
                        help="NPZ file produced by gtg.py (with 'prob', 'y', 'time_idx', 'x').")
    parser.add_argument("--rl-model", type=str,
                        default="runs_rl/PPO_GTG_RL/PPO_GTG_RL_final.zip",
                        help="Path to trained PPO model zip.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' or 'cuda'.")
    parser.add_argument("--total-years", type=float, default=20.0,
                        help="Total coverage horizon (years).")
    parser.add_argument("--gtg-train-years", type=float, default=12.0,
                        help="Years used for GTG train/val/test (first part).")
    parser.add_argument("--thr-gtg", type=float, default=0.53,
                        help="GTG probability threshold for baseline decision.")
    parser.add_argument("--thr-rl", type=float, default=0.50,
                        help="RL probability threshold for alert (on p(action=1)).")
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Max number of event examples to write to JSON.")
    parser.add_argument("--out", type=str, default="gtg_rl_explanations.json",
                        help="Output JSON file for explanations.")
    args = parser.parse_args()

    # ===== Load NPZ =====
    print(f"Loading RL NPZ from {args.npz} ...")
    data = np.load(args.npz, allow_pickle=True)

    prob = data["prob"].astype(np.float32)
    y = data["y"].astype(int)
    time_idx = data["time_idx"].astype(int)
    x = data["x"].astype(np.float32)

    # ===== Rebuild RL region and splits =====
    rl_mask, rl_train_mask, rl_val_mask, rl_test_mask = build_rl_region(
        time_idx=time_idx,
        total_years=args.total_years,
        gtg_train_years=args.gtg_train_years,
    )

    prob_rl = prob[rl_mask]
    y_rl = y[rl_mask]
    time_rl = time_idx[rl_mask]
    x_rl = x[rl_mask]

    global_idx_rl = np.where(rl_mask)[0]
    test_idx_rl_all = np.where(rl_test_mask)[0]       # indices in RL arrays
    global_test_idx_all = global_idx_rl[test_idx_rl_all]

    y_test_all = y_rl[test_idx_rl_all]
    prob_test_all = prob_rl[test_idx_rl_all]
    time_test_all = time_rl[test_idx_rl_all]
    x_test_all = x_rl[test_idx_rl_all]

    print(f"RL TEST nodes (all): {len(test_idx_rl_all)}")

    # ===== Focus on true events (y=1) in RL TEST =====
    event_mask = y_test_all == 1
    if not np.any(event_mask):
        print("No positive events in RL TEST subset; nothing to explain.")
        return

    test_idx_rl = test_idx_rl_all[event_mask]
    global_test_idx = global_test_idx_all[event_mask]
    y_test = y_test_all[event_mask]
    prob_test = prob_test_all[event_mask]
    time_test = time_test_all[event_mask]
    x_test = x_test_all[event_mask]

    print(f"RL TEST events (y=1) to explain: {len(test_idx_rl)}")

    # ===== Build observations exactly like GTGAnomalyEnv =====
    obs_list = []
    for p, feats in zip(prob_test, x_test):
        obs_list.append(np.concatenate([[p], feats], axis=0))
    obs = np.stack(obs_list, axis=0).astype(np.float32)

    # ===== Load RL model & get action probabilities =====
    print(f"Loading PPO model from {args.rl_model} on device '{args.device}' ...")
    model = PPO.load(args.rl_model, device=args.device)
    model.policy.to(args.device)
    model.policy.eval()

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, device=args.device)
        dist = model.policy.get_distribution(obs_tensor)
        # Categorical distribution over 2 actions: 0=ignore, 1=alert
        action_probs = dist.distribution.probs.detach().cpu().numpy()

    p_alert = action_probs[:, 1]  # p(action=1)
    rl_pred = (p_alert >= args.thr_rl).astype(int)
    gtg_pred = (prob_test >= args.thr_gtg).astype(int)

    # ===== Build explanation records =====
    examples = []
    for i in range(len(global_test_idx)):
        gidx = int(global_test_idx[i])
        rl_idx = int(test_idx_rl[i])
        t = int(time_test[i])
        label = int(y_test[i])  # always 1 here
        base_prob = float(prob_test[i])
        base_decision = int(gtg_pred[i])
        rl_prob = float(p_alert[i])
        rl_decision = int(rl_pred[i])

        # Effect of RL vs GTG for this event
        if base_decision == 0 and rl_decision == 1:
            effect = "rescued_event"        # GTG would miss, RL catches
        elif base_decision == 1 and rl_decision == 0:
            effect = "dropped_event"        # GTG would alert, RL silences
        elif base_decision == rl_decision == 1:
            effect = "agree_on_event"       # both alert
        else:
            effect = "both_missed"          # both say no alert (rare if thr/rl tuned)

        examples.append(
            dict(
                global_nid=gidx,
                rl_index=rl_idx,
                time_index=t,
                label=label,
                gtg_prob=base_prob,
                gtg_pred_at_thr=base_decision,
                rl_prob_alert=rl_prob,
                rl_pred_at_thr=rl_decision,
                effect=effect,
            )
        )

    # Sort by RL alert probability, descending
    examples.sort(key=lambda d: d["rl_prob_alert"], reverse=True)

    if args.max_examples is not None and len(examples) > args.max_examples:
        examples = examples[: args.max_examples]

    with open(args.out, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Wrote {len(examples)} RL explanations to {args.out}")


if __name__ == "__main__":
    main()
