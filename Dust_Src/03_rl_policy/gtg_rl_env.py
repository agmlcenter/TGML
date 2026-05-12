# gtg_rl_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Base rewards
    tp: float = 8.0
    fn: float = -35.0
    fp: float = -70.0
    tn: float = 0.0

    # Alert cost (usually <= 0)
    alert_cost: float = -1.5

    # If False, do NOT apply alert_cost on TP (only on FP)
    alert_cost_on_tp: bool = False

    # Mild state-dependent shaping using p_gtg in [0,1]
    miss_scale: float = 0.5   # scales TP/FN by p_gtg
    fp_scale: float = 1.0     # scales FP by p_gtg


class GTGAnomalyEnv(gym.Env):
    """
    RL environment for GTG-based anomaly decision-making.

    State: [p_gtg, x_0, x_1, ..., x_{F-1}]
    Action: Discrete(2)  (0: no alert, 1: alert)

    Training sampling (oversample_pos=True):
      - with probability p_pos -> sample positive
      - else, with probability p_hard_neg -> sample hard negative (if any)
      - else -> sample easy negative (fallbacks included)

    Evaluation sampling (oversample_pos=False):
      - sample according to real pos/neg ratio within this split
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_prob: np.ndarray,     # [N]
        x: np.ndarray,             # [N, F]
        y: np.ndarray,             # [N], {0,1}
        pos_idx: np.ndarray,       # indices of positives in THIS split
        neg_idx: np.ndarray,       # indices of negatives in THIS split
        reward_cfg: RewardConfig,
        max_steps_per_episode: int = 4096,
        oversample_pos: bool = True,
        p_pos: float = 0.2,
        p_hard_neg: float = 0.4,
        hard_neg_min_prob: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__()

        assert base_prob.shape[0] == x.shape[0] == y.shape[0]
        self.base_prob = base_prob.astype(np.float32)
        self.x = x.astype(np.float32)
        self.y = y.astype(int)

        self.pos_idx = np.array(pos_idx, dtype=int)
        self.neg_idx = np.array(neg_idx, dtype=int)

        self.reward_cfg = reward_cfg
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.oversample_pos = bool(oversample_pos)

        self.p_pos = float(p_pos)
        self.p_hard_neg = float(p_hard_neg)
        self.hard_neg_min_prob = float(hard_neg_min_prob)

        assert 0.0 <= self.p_pos <= 1.0
        assert 0.0 <= self.p_hard_neg <= 1.0
        assert self.p_pos + self.p_hard_neg <= 1.0 + 1e-6

        self.num_samples = self.base_prob.shape[0]
        self.obs_dim = self.x.shape[1] + 1  # p_gtg + features

        # RNG (important for SubprocVecEnv determinism)
        self._rng = np.random.default_rng(seed)

        # Hard/easy negative pools (for training only)
        if self.neg_idx.size > 0:
            neg_probs = self.base_prob[self.neg_idx]
            hard_mask = neg_probs >= self.hard_neg_min_prob
            self.hard_neg_idx = self.neg_idx[hard_mask]
            self.easy_neg_idx = self.neg_idx[~hard_mask]
        else:
            self.hard_neg_idx = np.array([], dtype=int)
            self.easy_neg_idx = np.array([], dtype=int)

        self.action_space = spaces.Discrete(2)

        # Observations are normalized-ish; keep broad bounds
        low = np.full((self.obs_dim,), -10.0, dtype=np.float32)
        high = np.full((self.obs_dim,), 10.0, dtype=np.float32)
        low[0] = 0.0
        high[0] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._step_count = 0
        self._current_idx = None

    def _sample_index(self) -> int:
        # Empty split fallback
        if self.pos_idx.size == 0 and self.neg_idx.size == 0:
            return int(self._rng.integers(0, self.num_samples))

        # If only one class exists, just sample it
        if self.pos_idx.size == 0:
            return int(self._rng.choice(self.neg_idx))
        if self.neg_idx.size == 0:
            return int(self._rng.choice(self.pos_idx))

        # ---- TRAINING: oversampled mixture ----
        if self.oversample_pos:
            r = float(self._rng.random())
            if r < self.p_pos:
                return int(self._rng.choice(self.pos_idx))

            r2 = float(self._rng.random())
            if r2 < self.p_hard_neg and self.hard_neg_idx.size > 0:
                return int(self._rng.choice(self.hard_neg_idx))

            if self.easy_neg_idx.size > 0:
                return int(self._rng.choice(self.easy_neg_idx))

            # fallback
            return int(self._rng.choice(self.neg_idx))

        # ---- EVAL: real distribution within this split ----
        p_real = self.pos_idx.size / (self.pos_idx.size + self.neg_idx.size)
        return int(self._rng.choice(self.pos_idx)) if float(self._rng.random()) < p_real else int(self._rng.choice(self.neg_idx))

    def _make_obs(self, idx: int) -> np.ndarray:
        p = self.base_prob[idx]
        feats = self.x[idx]
        return np.concatenate([[p], feats], axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._current_idx = self._sample_index()
        obs = self._make_obs(self._current_idx)
        return obs, {}

    def step(self, action: int):
        idx = int(self._current_idx)
        label = int(self.y[idx])
        p = float(self.base_prob[idx])  # p_gtg in [0,1]

        rc = self.reward_cfg

        # Base reward + mild shaping
        if label == 1:
            if action == 1:
                # Reward more when GTG was less confident (p small)
                scale = 1.0 + rc.miss_scale * (1.0 - p)
                reward = rc.tp * scale
            else:
                # Penalize more when GTG was more confident (p large)
                scale = 1.0 + rc.miss_scale * p
                reward = rc.fn * scale
        else:
            if action == 1:
                # Penalize more when you alarm on very low-p nodes
                scale = 1.0 + rc.fp_scale * (1.0 - p)
                reward = rc.fp * scale
            else:
                reward = rc.tn

        # Alert cost
        if action == 1:
            if (label == 0) or rc.alert_cost_on_tp:
                reward += rc.alert_cost

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps_per_episode

        self._current_idx = self._sample_index()
        obs = self._make_obs(self._current_idx)
        info = {"idx": idx, "label": label}

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None
