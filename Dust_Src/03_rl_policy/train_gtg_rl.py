# train_gtg_rl.py
import argparse
import os
import json
import math

import numpy as np
import torch
from torch import nn

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import obs_as_tensor

from gtg_rl_env import GTGAnomalyEnv, RewardConfig


# ----------------------------
# Early-warning metrics + threshold selection (same idea as GTG)
# ----------------------------
def compute_early_warning_stats(y_true, y_score, time_idx, thr, window_days=5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    time_idx = np.asarray(time_idx, dtype=int)

    event_idx = np.where(y_true == 1)[0]
    n_events = int(event_idx.size)

    alerts = (y_score >= thr)
    lead_times = []
    hits = 0

    if n_events == 0:
        neg_mask = (y_true == 0)
        n_neg = int(neg_mask.sum())
        fa_count = int(np.logical_and(alerts, neg_mask).sum()) if n_neg > 0 else 0
        fa_rate = (fa_count / n_neg) if n_neg > 0 else float("nan")
        return dict(
            events=0, hit_rate=float("nan"), miss_rate=float("nan"),
            mean_lead=float("nan"), median_lead=float("nan"),
            window_days=window_days, thr=float(thr),
            fa_rate=float(fa_rate), fa_count=int(fa_count), fa_per_event=float("nan")
        )

    for e in event_idx:
        t_event = time_idx[e]
        t_min = t_event - window_days
        in_window = (time_idx >= t_min) & (time_idx <= t_event)
        window_alerts = alerts & in_window
        if np.any(window_alerts):
            hits += 1
            t_first = time_idx[window_alerts].min()
            lead_times.append(float(t_event - t_first))

    hit_rate = hits / n_events
    miss_rate = 1.0 - hit_rate
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0

    neg_mask = (y_true == 0)
    n_neg = int(neg_mask.sum())
    fa_count = int(np.logical_and(alerts, neg_mask).sum()) if n_neg > 0 else 0
    fa_rate = (fa_count / n_neg) if n_neg > 0 else 0.0
    fa_per_event = fa_count / n_events if n_events > 0 else float("nan")

    return dict(
        events=n_events,
        hit_rate=float(hit_rate),
        miss_rate=float(miss_rate),
        mean_lead=float(mean_lead),
        median_lead=float(median_lead),
        window_days=int(window_days),
        thr=float(thr),
        fa_rate=float(fa_rate),
        fa_count=int(fa_count),
        fa_per_event=float(fa_per_event),
    )


def find_low_fa_threshold(y_true_val, y_score_val, time_val, window_days=5, target_hit=0.70, n_thresholds=1001):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    best_thr = 0.5
    best_hit = -1.0
    best_fa_per_event = float("inf")
    best_fa_rate = 1.0

    fallback_thr = 0.5
    fallback_hit = -1.0
    fallback_fa_per_event = float("inf")
    fallback_fa_rate = 1.0

    for thr in thresholds:
        ew = compute_early_warning_stats(y_true_val, y_score_val, time_val, thr, window_days)
        hit = ew["hit_rate"]
        fa_rate = ew["fa_rate"]
        fa_per_event = ew["fa_per_event"]

        if hit > fallback_hit or (math.isclose(hit, fallback_hit) and fa_per_event < fallback_fa_per_event):
            fallback_hit = hit
            fallback_fa_per_event = fa_per_event
            fallback_fa_rate = fa_rate
            fallback_thr = thr

        if hit >= target_hit:
            if fa_per_event < best_fa_per_event:
                best_thr = thr
                best_hit = hit
                best_fa_rate = fa_rate
                best_fa_per_event = fa_per_event

    if best_fa_per_event == float("inf"):
        best_thr = fallback_thr
        best_hit = fallback_hit
        best_fa_rate = fallback_fa_rate
        best_fa_per_event = fallback_fa_per_event

    return dict(thr=float(best_thr), hit_rate=float(best_hit), fa_rate=float(best_fa_rate), fa_per_event=float(best_fa_per_event))


@torch.no_grad()
def policy_p_alert(model: PPO, obs_np: np.ndarray) -> np.ndarray:
    obs_t = obs_as_tensor(obs_np, model.device)
    dist = model.policy.get_distribution(obs_t)
    probs = dist.distribution.probs  # [B,2] categorical
    return probs[:, 1].detach().cpu().numpy()


def make_env_fn(
    base_prob, x, y, pos_idx, neg_idx, reward_cfg, max_steps_per_episode,
    oversample_pos, p_pos, p_hard_neg, hard_neg_min_prob, seed
):
    def _init():
        env = GTGAnomalyEnv(
            base_prob=base_prob,
            x=x,
            y=y,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            reward_cfg=reward_cfg,
            max_steps_per_episode=max_steps_per_episode,
            oversample_pos=oversample_pos,
            p_pos=p_pos,
            p_hard_neg=p_hard_neg,
            hard_neg_min_prob=hard_neg_min_prob,
            seed=seed,
        )
        env = Monitor(env)
        env = FlattenObservation(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="gtg_tgcn_rl_data.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-years", type=float, default=20.0)
    parser.add_argument("--gtg-train-years", type=float, default=12.0)

    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)

    parser.add_argument("--max-ep-steps-train", type=int, default=4096)
    parser.add_argument("--max-ep-steps-eval", type=int, default=4096)

    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--target-hit", type=float, default=0.70)
    parser.add_argument("--n-thresholds", type=int, default=1001)
    parser.add_argument("--logdir", type=str, default="runs_rl")

    # Oversampling knobs (TRAIN only)
    parser.add_argument("--p-pos", type=float, default=0.20)
    parser.add_argument("--p-hard-neg", type=float, default=0.40)
    parser.add_argument("--hard-neg-min-prob", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    print(f"Loading RL data from {args.npz} ...")
    data = np.load(args.npz, allow_pickle=True)

    prob = data["prob"].astype(np.float32)
    y = data["y"].astype(int)
    time_idx = data["time_idx"].astype(int)
    x = data["x"].astype(np.float32)

    # ==== Select RL region: LAST (total-years - gtg-train-years) years ====
    T_total = int(time_idx.max() + 1)
    T_cut = int(T_total * args.gtg_train_years / args.total_years)

    rl_mask = time_idx >= T_cut
    prob_rl = prob[rl_mask]
    y_rl = y[rl_mask]
    time_rl = time_idx[rl_mask]
    x_rl = x[rl_mask]

    print(f"Total time steps: {T_total}")
    print(f"GTG train end index (T_cut): {T_cut}")
    print(f"RL region nodes: {prob_rl.shape[0]}")

    # ==== Time-based splits inside RL region (train/val/test) ====
    t0 = int(time_rl.min())
    Trl = int(time_rl.max() - t0 + 1)

    train_T = t0 + int(0.70 * Trl)
    val_T = t0 + int(0.85 * Trl)

    rl_train_mask = time_rl < train_T
    rl_val_mask = (time_rl >= train_T) & (time_rl < val_T)
    rl_test_mask = time_rl >= val_T

    idx_train = np.where(rl_train_mask)[0]
    idx_val = np.where(rl_val_mask)[0]
    idx_test = np.where(rl_test_mask)[0]

    print(f"RL train nodes: {idx_train.size}")
    print(f"RL val   nodes: {idx_val.size}")
    print(f"RL test  nodes: {idx_test.size}")

    # ==== Positive/negative indices per split (indices are in [0..N_rl-1]) ====
    y_train = y_rl[idx_train]
    y_val = y_rl[idx_val]
    y_test = y_rl[idx_test]

    train_pos_idx = idx_train[y_train == 1]
    train_neg_idx = idx_train[y_train == 0]

    val_pos_idx = idx_val[y_val == 1]
    val_neg_idx = idx_val[y_val == 0]

    test_pos_idx = idx_test[y_test == 1]
    test_neg_idx = idx_test[y_test == 0]

    print(f"Train positives: {train_pos_idx.size}, negatives: {train_neg_idx.size}")
    print(f"Val   positives: {val_pos_idx.size}, negatives: {val_neg_idx.size}")
    print(f"Test  positives: {test_pos_idx.size}, negatives: {test_neg_idx.size}")

    # ==== Middle-ground reward (start here) ====
    reward_cfg = RewardConfig(
        tp=8.0,
        fn=-35.0,
        fp=-70.0,
        tn=0.0,
        alert_cost=-1.5,
        alert_cost_on_tp=False,
        miss_scale=0.5,
        fp_scale=1.0,
    )

    # ==== Vectorized training envs ====
    train_env_fns = []
    for i in range(args.n_envs):
        train_env_fns.append(
            make_env_fn(
                base_prob=prob_rl,
                x=x_rl,
                y=y_rl,
                pos_idx=train_pos_idx,
                neg_idx=train_neg_idx,
                reward_cfg=reward_cfg,
                max_steps_per_episode=args.max_ep_steps_train,
                oversample_pos=True,
                p_pos=args.p_pos,
                p_hard_neg=args.p_hard_neg,
                hard_neg_min_prob=args.hard_neg_min_prob,
                seed=args.seed + i,
            )
        )
    train_env = SubprocVecEnv(train_env_fns)

    # ==== Eval env (real distribution within VAL split) ====
    eval_env = DummyVecEnv([
        make_env_fn(
            base_prob=prob_rl,
            x=x_rl,
            y=y_rl,
            pos_idx=val_pos_idx,
            neg_idx=val_neg_idx,
            reward_cfg=reward_cfg,
            max_steps_per_episode=args.max_ep_steps_eval,
            oversample_pos=False,
            p_pos=0.0,
            p_hard_neg=0.0,
            hard_neg_min_prob=args.hard_neg_min_prob,
            seed=args.seed + 10_000,
        )
    ])

    # ==== PPO model ====
    log_path = os.path.join(args.logdir, "PPO_GTG_RL")
    print(f"Logging to {log_path}")

    total_batch = args.n_steps * args.n_envs
    batch_size = 8192 if total_batch >= 8192 else max(256, (total_batch // 4) // 256 * 256)

    model = PPO(
        "MlpPolicy",
        train_env,
        device=args.device,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=30,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=nn.Tanh),
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=os.path.join(log_path, "eval"),
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    final_path = os.path.join(log_path, "PPO_GTG_RL_final.zip")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    # ----------------------------
    # Offline threshold tuning on VAL using p(alert)
    # ----------------------------
    obs_val = np.concatenate([prob_rl[idx_val, None], x_rl[idx_val]], axis=1).astype(np.float32)
    score_val = policy_p_alert(model, obs_val)

    lowfa = find_low_fa_threshold(
        y_true_val=y_rl[idx_val],
        y_score_val=score_val,
        time_val=time_rl[idx_val],
        window_days=args.window_days,
        target_hit=args.target_hit,
        n_thresholds=args.n_thresholds,
    )
    print(
        f"[RL low-FA thr on VAL] thr={lowfa['thr']:.4f} "
        f"hit_rate_val={lowfa['hit_rate']:.3f} FA_rate_val={lowfa['fa_rate']:.5f} "
        f"FA_per_event_val={lowfa['fa_per_event']:.2f}"
    )

    thr_path = os.path.join(log_path, "chosen_threshold.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(window_days=args.window_days, target_hit=args.target_hit, **lowfa),
            f,
            indent=2
        )
    print(f"Saved chosen threshold to: {thr_path}")

    # ----------------------------
    # Evaluate on TEST at chosen threshold
    # ----------------------------
    thr = lowfa["thr"]
    obs_test = np.concatenate([prob_rl[idx_test, None], x_rl[idx_test]], axis=1).astype(np.float32)
    score_test = policy_p_alert(model, obs_test)

    y_pred = (score_test >= thr).astype(int)
    y_true = y_rl[idx_test].astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    print("\n=== RL PPO Evaluation on RL TEST subset (node-level) ===")
    print(f"thr(p_alert): {thr:.4f}")
    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1-score:  {f1:.6f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array([[tn, fp], [fn, tp]], dtype=int))

    ew = compute_early_warning_stats(
        y_true=y_true,
        y_score=score_test,
        time_idx=time_rl[idx_test],
        thr=thr,
        window_days=args.window_days,
    )
    print(f"\n=== RL PPO Early-warning metrics ({args.window_days}-day window, thr={thr:.4f}) ===")
    print(f"Events (y=1)        : {ew['events']}")
    print(f"Hit rate            : {ew['hit_rate']:.3f}")
    print(f"Miss rate           : {ew['miss_rate']:.3f}")
    print(f"Mean lead (days)    : {ew['mean_lead']:.2f}")
    print(f"Median lead (days)  : {ew['median_lead']:.2f}")
    print(f"False-alarm rate    : {ew['fa_rate']:.5f}")
    print(f"False alarms total  : {ew['fa_count']}")
    print(f"False alarms / event: {ew['fa_per_event']:.2f}")


if __name__ == "__main__":
    main()
