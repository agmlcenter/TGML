# activation_daily_pipeline_v6.py
#
# Daily GTG–TGCN pipeline for dust source activation + ablation runs:
# - Daily windows (no weekly aggregation)
# - NP thresholding + calibration
# - High-wind conditional metrics
# - Ranking@K on high-wind days
# - Simple node-event metrics (miss rate / lead time)
# - Gradient×input time–lag explainability (CPU-safe)
# - Ablations over model/loss configurations (10 epochs each)

import os, json, time, logging, math, random, sys, gc, traceback
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Iterable, Sequence

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, f1_score, brier_score_loss, precision_score
)

_HAS_SCIPY = True
try:
    from scipy.stats import genpareto
except Exception:
    _HAS_SCIPY = False

SEED = 33
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    torch.set_float32_matmul_precision('high')
except Exception as e:
    print("ERROR:", e, file=sys.stderr, flush=True)

def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb, file=sys.stderr)
    sys.stderr.flush()

sys.excepthook = excepthook

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger("activation.daily.v6")

# -------------------------------------------------------------------
# Basic utils, loading, preprocessing
# -------------------------------------------------------------------
def _map_node_to_id(s: str, grid_size: int) -> int:
    a, b = s.split('_')
    return int(a) * grid_size + int(b)


def _id_to_rc(nid: int, grid_size: int) -> Tuple[int, int]:
    return nid // grid_size, nid % grid_size


def load_daily(csv_path: str, grid_size: int = 20) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.sort_values(['node', 'date'], inplace=True)
    df['node_id'] = df['node'].apply(lambda s: _map_node_to_id(s, grid_size)).astype(np.int32)
    return df


def impute_missing_per_node(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if df[c].dtype.kind in 'fc' and c not in ['label', 'node_id']]
    for _, g in df.groupby('node', sort=False):
        idx = g.index
        vals = g[num_cols].ffill().bfill()
        df.loc[idx, num_cols] = vals.fillna(vals.mean())
    return df


def add_static_priors(df: pd.DataFrame, grid_size: int = 20) -> pd.DataFrame:
    """
    Add VISI-proximity priors if VISI hotspot columns exist, else zeros.
    """
    hotspot_cols = [c for c in df.columns if c.lower() in ('visi_hotspot', 'visi', 'hds', 'is_visi')]
    if hotspot_cols:
        tag = hotspot_cols[0]
        hs_nodes = set(df.loc[df[tag].astype(int) == 1, 'node_id'].astype(int).unique().tolist())
        if hs_nodes:
            nid_to_minL1: Dict[int, int] = {}
            all_coords = {nid: _id_to_rc(nid, grid_size) for nid in df['node_id'].unique()}
            hs_coords = [_id_to_rc(n, grid_size) for n in hs_nodes]
            for nid, (r, c) in all_coords.items():
                d = min(abs(r - hr) + abs(c - hc) for hr, hc in hs_coords)
                nid_to_minL1[nid] = d
            prox = df['node_id'].map(lambda nid: nid_to_minL1.get(int(nid), grid_size)).astype('float32')
            df['visi_proximity'] = prox
            df['visi_proximity_inv'] = 1.0 / (1.0 + df['visi_proximity'])
        else:
            df['visi_proximity'] = 0.0
            df['visi_proximity_inv'] = 0.0
    else:
        df['visi_proximity'] = 0.0
        df['visi_proximity_inv'] = 0.0
    return df

# -------------------------------------------------------------------
# Scaling, daily windows, index
# -------------------------------------------------------------------
def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
):
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].astype('float32'))
    val_df[feature_cols] = scaler.transform(val_df[feature_cols].astype('float32'))
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].astype('float32'))
    return train_df, val_df, test_df, scaler


def ensure_node_and_date(df: pd.DataFrame, grid_size: int = 20) -> pd.DataFrame:
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("Expected 'date' column for daily pipeline.")
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    if 'node_id' not in df.columns:
        if 'node' in df.columns:
            df['node_id'] = df['node'].apply(lambda s: _map_node_to_id(s, grid_size)).astype(np.int32)
        else:
            raise KeyError("Neither 'node_id' nor 'node' present")

    if 'node' not in df.columns:
        def _id_to_node(nid: int) -> str:
            r = int(nid) // grid_size
            c = int(nid) % grid_size
            return f"{r}_{c}"
        df['node'] = df['node_id'].apply(_id_to_node)

    return df


def create_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    prox_col: Optional[str],
    window_size: int,
    out_dir: str,
    suffix: str,
    min_window: int = 32,
    drop_short_nodes: bool = True
):
    """
    Sliding history windows per node.
    df must have columns: node (or node_id), date, label, node_id, feature_cols, prox_col(optional).
    """
    os.makedirs(out_dir, exist_ok=True)
    node_key = 'node' if 'node' in df.columns else ('node_id' if 'node_id' in df.columns else None)
    if node_key is None:
        raise KeyError("create_windows expects 'node' or 'node_id'")
    df = df.sort_values([node_key, 'date']).reset_index(drop=True)

    if drop_short_nodes:
        sizes = df.groupby(node_key, sort=False)['date'].size().astype(int)
        keep_nodes = set(sizes[sizes >= max(min_window, window_size + 1)].index.tolist())
        df = df[df[node_key].isin(keep_nodes)].copy()
        eff_win = int(window_size)
    else:
        max_len = int(df.groupby(node_key, sort=False)['date'].size().max()) if len(df) else 0
        eff_win = int(min(window_size, max(1, max_len - 1)))

    total = 0
    for _, g in df.groupby(node_key, sort=False):
        total += max(0, len(g) - eff_win)
    if total == 0:
        raise RuntimeError(
            f"No windows produced. Reduce window_size or disable drop_short_nodes. (eff_win={eff_win})"
        )

    nfeat = len(feature_cols)
    seq_mm = np.memmap(f'{out_dir}/sequences{suffix}.dat', 'float32', 'w+', shape=(total, eff_win, nfeat))
    lbl_mm = np.memmap(f'{out_dir}/labels{suffix}.dat', 'float32', 'w+', shape=(total,))
    nid_mm = np.memmap(f'{out_dir}/node_ids{suffix}.dat', 'int32', 'w+', shape=(total,))
    dat_mm = np.memmap(f'{out_dir}/dates_int64{suffix}.dat', 'int64', 'w+', shape=(total,))
    prx_mm = np.memmap(f'{out_dir}/prox{suffix}.dat', 'float32', 'w+', shape=(total,))

    idx = 0
    for _, g in df.groupby(node_key, sort=False):
        g = g.reset_index(drop=True)
        X = g[feature_cols].astype('float32').values  # [L, F]
        prox_arr = g[prox_col].astype('float32').values if prox_col else np.zeros(len(g), dtype=np.float32)
        y = g['label'].astype('float32').values
        nid = g['node_id'].astype('int32').values
        tns = pd.to_datetime(g['date']).values.astype('datetime64[ns]').astype('int64')
        L = len(g)
        if L <= eff_win:
            continue
        for i in range(eff_win, L):
            seq_mm[idx] = X[i - eff_win:i]
            lbl_mm[idx] = y[i]
            nid_mm[idx] = nid[i]
            dat_mm[idx] = tns[i]
            prx_mm[idx] = prox_arr[i]
            idx += 1

    seq_mm.flush()
    lbl_mm.flush()
    nid_mm.flush()
    dat_mm.flush()
    prx_mm.flush()

    seq = np.memmap(f'{out_dir}/sequences{suffix}.dat', 'float32', 'r', shape=(total, eff_win, nfeat))
    lbl = np.memmap(f'{out_dir}/labels{suffix}.dat', 'float32', 'r', shape=(total,))
    nid = np.memmap(f'{out_dir}/node_ids{suffix}.dat', 'int32', 'r', shape=(total,))
    dat = np.memmap(f'{out_dir}/dates_int64{suffix}.dat', 'int64', 'r', shape=(total,))
    prx = np.memmap(f'{out_dir}/prox{suffix}.dat', 'float32', 'r', shape=(total,))
    return seq, lbl, nid, dat, prx, eff_win


def build_snapshot_index(dates_int64, grid_size=20):
    """
    Build snapshot index and static grid adjacency.
    """
    n = grid_size * grid_size
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if i > 0:
                adj[idx, idx - grid_size] = 1
            if i < grid_size - 1:
                adj[idx, idx + grid_size] = 1
            if j > 0:
                adj[idx, idx - 1] = 1
            if j < grid_size - 1:
                adj[idx, idx + 1] = 1
    edge_index, _ = dense_to_sparse(torch.tensor(adj, dtype=torch.uint8))
    edge_index = edge_index.long().contiguous()

    dates = np.asarray(dates_int64)
    uniq_dates, inv = np.unique(dates, return_inverse=True)
    idx_lists = [np.where(inv == t)[0].astype(np.int32) for t in range(len(uniq_dates))]
    return uniq_dates, idx_lists, edge_index

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class SnapshotDataset(Dataset):
    def __init__(self, sequences_mm, labels_mm, node_ids_mm, prox_mm, dates_uniq, idx_lists, edge_index):
        self.seqs = sequences_mm
        self.lbls = labels_mm
        self.nids = node_ids_mm
        self.prox = prox_mm
        self.dates_uniq = dates_uniq
        self.idx_lists = idx_lists
        self.edge_index = edge_index
        self._cache = OrderedDict()
        self._cap = 512
        self.feature_names: Optional[List[str]] = None  # will be set from outside

    def __getitem__(self, t):
        if t in self._cache:
            d = self._cache[t]
            return Data(
                x=d.x.clone(),
                y=d.y.clone(),
                edge_index=d.edge_index.clone(),
                nids=d.nids.clone(),
                prox=d.prox.clone(),
            )
        idxs = self.idx_lists[t]
        x = torch.from_numpy(np.array(self.seqs[idxs], dtype=np.float32))      # [N, T, F]
        y = torch.from_numpy(np.array(self.lbls[idxs], dtype=np.float32))      # [N]
        nids = torch.from_numpy(np.array(self.nids[idxs], dtype=np.int32))     # [N]
        prox = torch.from_numpy(np.array(self.prox[idxs], dtype=np.float32))   # [N]
        data = Data(x=x, y=y, edge_index=self.edge_index, nids=nids, prox=prox)
        self._cache[t] = data
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return data

    def __len__(self):
        return len(self.idx_lists)

# -------------------------------------------------------------------
# Subgraph construction
# -------------------------------------------------------------------
def make_edge_index_sub(edge_index: torch.Tensor, nids_sub: torch.Tensor, grid_size: int) -> torch.Tensor:
    dev = edge_index.device
    nids = nids_sub.long().to(dev)
    nids = nids[nids >= 0]
    if nids.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=dev)

    max_e = int(edge_index.max().item()) if edge_index.numel() > 0 else -1
    max_n = int(nids.max().item())
    N_total = max(max_e, max_n) + 1

    src, dst = edge_index[0].to(dev), edge_index[1].to(dev)
    in_bounds = (src >= 0) & (src < N_total) & (dst >= 0) & (dst < N_total)
    src = src[in_bounds]
    dst = dst[in_bounds]

    present = torch.zeros(N_total, dtype=torch.bool, device=dev)
    present[nids] = True
    mask = present[src] & present[dst]
    src = src[mask]
    dst = dst[mask]

    if src.numel() == 0:
        n = nids.numel()
        idx = torch.arange(n, device=dev, dtype=torch.long)
        return torch.stack([idx, idx], dim=0).contiguous()

    remap = torch.full((N_total,), -1, dtype=torch.long, device=dev)
    remap[nids] = torch.arange(nids.numel(), device=dev, dtype=torch.long)
    ei_sub = torch.stack([remap[src], remap[dst]], dim=0)
    ok = (ei_sub[0] >= 0) & (ei_sub[1] >= 0)
    ei_sub = ei_sub[:, ok]
    if ei_sub.numel() == 0:
        n = nids.numel()
        idx = torch.arange(n, device=dev, dtype=torch.long)
        ei_sub = torch.stack([idx, idx], dim=0)
    return ei_sub.contiguous()

# -------------------------------------------------------------------
# Group Temporal Gate (daily)
# -------------------------------------------------------------------
class GroupTemporalGateDaily(nn.Module):
    def __init__(self, lookback_steps: int, feat_dim: int):
        super().__init__()
        self.L = lookback_steps
        self.groups = {
            0: dict(max_lags=self.L,               stride=1),
            1: dict(max_lags=min(4 * 30, self.L),  stride=4),
            2: dict(max_lags=min(90 * 2, self.L),  stride=30),
            3: dict(max_lags=min(365 * 3, self.L), stride=90),
        }
        self.feat2group = nn.Parameter(
            torch.tensor([i % len(self.groups) for i in range(feat_dim)], dtype=torch.long),
            requires_grad=False
        )
        self.logits = nn.ParameterDict({str(g): nn.Parameter(torch.zeros(self.L)) for g in self.groups})
        self.alpha = nn.ParameterDict({str(g): nn.Parameter(torch.tensor(0.0)) for g in self.groups})

    def _mask(self, gid, device):
        conf = self.groups[gid]
        m = torch.zeros(self.L, dtype=torch.bool, device=device)
        for i in range(0, conf['max_lags'], conf['stride']):
            if i < self.L:
                m[i] = True
        return m

    def _soft(self, logits, gid, device):
        m = self._mask(gid, device)
        v = torch.where(m, logits.to(device), torch.full_like(logits, -1e4))
        return torch.softmax(v, dim=0)

    def forward(self, X):
        """
        X: [B, T, N, F]
        """
        B, T, N, F = X.shape
        dev = X.device
        Xbnf = X.permute(0, 2, 3, 1).contiguous()  # [B, N, F, T]
        out = Xbnf.clone()
        for gid in self.groups:
            feats = (self.feat2group == gid).nonzero(as_tuple=True)[0].to(dev)
            if feats.numel() == 0:
                continue
            w = self._soft(self.logits[str(gid)], gid, dev)  # [L]
            kernel = torch.flip(w, dims=[0]).view(1, 1, -1)  # [1,1,L]
            Fg = feats.numel()
            weight = kernel.repeat(Fg, 1, 1)  # [Fg,1,L]
            Xg = Xbnf[:, :, feats, :].reshape(B * N, Fg, T)
            ctx = nn.functional.conv1d(Xg, weight=weight, stride=1, padding=self.L - 1, groups=Fg)[:, :, :T]
            a = torch.sigmoid(self.alpha[str(gid)])
            out[:, :, feats, :] = ((1 - a) * Xg + a * ctx).reshape(B, N, Fg, T)
        return out.permute(0, 3, 1, 2).contiguous()

    def entropy_reg(self, device):
        reg = 0.0
        for gid in self.groups:
            w = self._soft(self.logits[str(gid)], gid, device)
            p = w.clamp_min(1e-9)
            reg += 1e-3 * (-(p * p.log()).sum())
        return reg

# -------------------------------------------------------------------
# Models: TGCN + GRU-only
# -------------------------------------------------------------------
class TGCNContrast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 256,
        dropout: float = 0.3,
        lookback_steps: int = 180,
        use_gtg: bool = True
    ):
        super().__init__()
        self.lookback = lookback_steps
        self.use_gtg = use_gtg

        self.gtg = GroupTemporalGateDaily(self.lookback, in_channels)
        self.tem_attn = nn.MultiheadAttention(in_channels, num_heads=1, batch_first=True, dropout=dropout)
        self.gru = nn.GRU(in_channels, hidden, batch_first=True, num_layers=2, dropout=dropout)
        self.gcn1 = pyg_nn.GCNConv(hidden, hidden)
        self.gcn2 = pyg_nn.GCNConv(hidden, hidden)
        self.fc_cls = nn.Linear(hidden, 1)
        self.fc_contrast = nn.Linear(hidden, 1)
        self.beta_prox = nn.Parameter(torch.tensor(0.0))
        self.drop = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge = data.x.float(), data.edge_index  # x: [N, T, F]

        # reshape for GTG: [N, T, F] -> [1, T, N, F]
        x_b = x.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        # apply GTG (optional)
        if self.use_gtg:
            x_b = self.gtg(x_b)  # [1, T, N, F]

        # reshape back to [N, T, F]
        x = x_b.squeeze(0).permute(1, 0, 2).contiguous()

        # temporal attention + GRU
        x_attn, _ = self.tem_attn(x, x, x)  # [N, T, F]
        x = x + self.drop(x_attn)
        h, _ = self.gru(x.float())          # [N, T, H]
        h = h[:, -1, :]                     # [N, H]

        # spatial GCN
        h = torch.relu(self.gcn1(h, edge))
        h = self.drop(h)
        h = self.gcn2(h, edge)
        h = self.drop(h)

        main_logit = self.fc_cls(h).squeeze(-1)
        contrast = self.fc_contrast(h).squeeze(-1)

        if hasattr(data, 'prox'):
            main_logit = main_logit + self.beta_prox * data.prox.to(main_logit.dtype)

        return main_logit, contrast


class GRUOnly(nn.Module):
    """
    Simpler baseline: temporal attention + GRU + linear head, no GTG, no GCN.
    """
    def __init__(self, in_channels: int, hidden: int = 256, dropout: float = 0.3,
                 lookback_steps: int = 180):
        super().__init__()
        self.lookback = lookback_steps
        self.tem_attn = nn.MultiheadAttention(in_channels, num_heads=1,
                                              batch_first=True, dropout=dropout)
        self.gru = nn.GRU(in_channels, hidden, batch_first=True,
                          num_layers=2, dropout=dropout)
        self.fc_cls = nn.Linear(hidden, 1)
        self.beta_prox = nn.Parameter(torch.tensor(0.0))
        self.drop = nn.Dropout(dropout)

    def forward(self, data: Data):
        x = data.x.float()  # [N, T, F], edge_index ignored

        x_attn, _ = self.tem_attn(x, x, x)  # [N, T, F]
        x = x + self.drop(x_attn)
        h, _ = self.gru(x)                  # [N, T, H]
        h = h[:, -1, :]                     # [N, H]
        h = self.drop(h)
        main_logit = self.fc_cls(h).squeeze(-1)

        if hasattr(data, 'prox'):
            main_logit = main_logit + self.beta_prox * data.prox.to(main_logit.dtype)

        contrast = torch.zeros_like(main_logit)
        return main_logit, contrast

# -------------------------------------------------------------------
# Losses + mining
# -------------------------------------------------------------------
class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, weights=None):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = (1 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            a = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = a * loss
        if weights is not None:
            loss = loss * weights
        return loss.mean()


class BCEWithLogitsWeighted(nn.Module):
    """
    Standard BCEWithLogits with optional global pos_weight and per-sample weights.
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets, weights=None):
        if self.pos_weight is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
        if weights is not None:
            loss = loss * weights
        return loss.mean()


def pairwise_topk_push_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 0.2):
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.tensor(0.0, device=pos_scores.device)
    diffs = margin - (pos_scores.view(-1, 1) - neg_scores.view(1, -1))
    return torch.relu(diffs).mean()


def select_hard_negatives_by_percent(logits: torch.Tensor, y: torch.Tensor, top_percent: float = 0.02):
    with torch.no_grad():
        neg_idx = (y < 0.5).nonzero(as_tuple=True)[0]
        if neg_idx.numel() == 0:
            return neg_idx
        k = max(1, int(math.ceil(top_percent * neg_idx.numel())))
        scores = logits[neg_idx]
        topk = torch.topk(scores, k=min(k, scores.numel()), largest=True).indices
        return neg_idx[topk]

# -------------------------------------------------------------------
# Simple NMS in 2D grid per day
# -------------------------------------------------------------------
def postprocess_3d_nms(scores: np.ndarray, nids: np.ndarray, days: np.ndarray,
                        grid_size: int, thr: float) -> np.ndarray:
    """
    Spatial non-max suppression per day:
    - threshold at thr
    - keep only local maxima vs 8-neighborhood
    """
    scores = np.asarray(scores)
    nids = np.asarray(nids, dtype=np.int64)
    days = np.asarray(days).astype('datetime64[D]')
    pred = np.zeros_like(scores, dtype=np.int32)

    df = pd.DataFrame({
        "score": scores,
        "nid": nids,
        "day": days
    })

    for d, g in df.groupby('day'):
        if g.empty:
            continue
        grid = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
        idx_map: Dict[Tuple[int, int], int] = {}
        for i_row, (nid, s) in g[['nid', 'score']].iterrows():
            r = int(nid) // grid_size
            c = int(nid) % grid_size
            grid[r, c] = float(s)
            idx_map[(r, c)] = i_row

        keep_idx = []
        for (r, c), i_global in idx_map.items():
            val = grid[r, c]
            if val < thr:
                continue
            r0, r1 = max(0, r - 1), min(grid_size - 1, r + 1)
            c0, c1 = max(0, c - 1), min(grid_size - 1, c + 1)
            neighborhood = grid[r0:r1 + 1, c0:c1 + 1]
            if val >= np.max(neighborhood):
                keep_idx.append(i_global)

        if keep_idx:
            pred[np.array(keep_idx, dtype=int)] = 1

    return pred

# -------------------------------------------------------------------
# PR@K helper (per day)
# -------------------------------------------------------------------
def weekly_PR_at_K(scores: np.ndarray, labels: np.ndarray, dates: np.ndarray, K_list=(10, 20, 50, 100)):
    resP = {}
    resR = {}
    df = pd.DataFrame({'score': scores, 'label': labels, 'date': dates})
    for K in K_list:
        tp = 0
        fp = 0
        pos = 0
        for _, g in df.groupby('date'):
            gg = g.sort_values('score', ascending=False).head(K)
            tp += int(gg['label'].sum())
            fp += int(len(gg) - gg['label'].sum())
            pos += int(g['label'].sum())
        resP[int(K)] = tp / max(1, (tp + fp))
        resR[int(K)] = tp / max(1, pos) if pos > 0 else 0.0
    return resP, resR

# -------------------------------------------------------------------
# Train loop (generalized for ablations)
# -------------------------------------------------------------------
def train(
    model,
    train_ds: SnapshotDataset,
    epochs: int = 10,
    lr: float = 7e-4,
    batch_size: int = 1,
    grid_size: int = 20,
    loss_type: str = "focal",         # "focal" or "bce"
    f_alpha=None,
    gamma: float = 3.0,
    rank_margin: float = 0.3,
    rank_weight: float = 0.8,         # weight on pairwise ranking loss
    target_pos_ratio: float = 0.3,
    num_workers: int = 0,
    top_neg_percent: float = 0.02,
    lambda_prev: float = 0.3,
    feat_names: Optional[List[str]] = None,
    pos_weight: Optional[float] = None,   # for BCE
):
    g = torch.Generator(device='cpu').manual_seed(SEED)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=False
    )

    # choose loss
    if loss_type == "focal":
        criterion = WeightedFocalLoss(gamma=gamma, alpha=f_alpha)
    elif loss_type == "bce":
        pw_tensor = None
        if pos_weight is not None:
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
        criterion = BCEWithLogitsWeighted(pos_weight=pw_tensor)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=2e-5,
        betas=(0.9, 0.98)
    )
    model.to(DEVICE)
    logger.info("Train start")

    idx_prev = None
    if feat_names:
        for i, nm in enumerate(feat_names):
            if 'prev4w_density' in nm.lower():
                idx_prev = i
                break

    for e in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        bcnt = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            x, y, nids = batch.x, batch.y, batch.nids

            pos_idx = (y > 0.5).nonzero(as_tuple=True)[0]
            neg_idx = (y < 0.5).nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0:
                keep_idx = neg_idx[
                    torch.randperm(neg_idx.numel(), device=y.device)[:min(1024, neg_idx.numel())]
                ]
            else:
                sample_neg = neg_idx[
                    torch.randperm(neg_idx.numel(), device=y.device)[
                        :min(20 * pos_idx.numel(), neg_idx.numel())
                    ]
                ]
                keep_idx = torch.unique(torch.cat([pos_idx, sample_neg], 0), sorted=False)

            if pos_idx.numel() > 0:
                cur_pos = (y[keep_idx] > 0.5).sum().item()
                cur_neg = int(keep_idx.numel() - cur_pos)
                need_pos = int(max(0, target_pos_ratio * (cur_pos + cur_neg) - cur_pos))
                if need_pos > 0:
                    rep = pos_idx[torch.randint(0, pos_idx.numel(), (need_pos,), device=y.device)]
                    keep_idx = torch.cat([keep_idx, rep], 0)

            x_sub = x[keep_idx]
            y_sub = y[keep_idx]
            nids_sub = nids[keep_idx]
            prox_sub = batch.prox[keep_idx]

            edge_index_sub = make_edge_index_sub(batch.edge_index, nids_sub, grid_size)

            opt.zero_grad(set_to_none=True)

            if edge_index_sub.numel() == 0 or x_sub.size(0) < 2:
                logits = torch.zeros_like(y_sub)
                loss = criterion(logits, y_sub.float())
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                bcnt += 1
                continue

            logits, contrast = model(
                Data(
                    x=x_sub.float(),
                    y=y_sub.float(),
                    edge_index=edge_index_sub,
                    nids=nids_sub,
                    prox=prox_sub
                )
            )

            loss_push = torch.tensor(0.0, device=logits.device)
            if rank_weight > 0.0:
                hard_negs = select_hard_negatives_by_percent(
                    logits.detach(), y_sub, top_percent=top_neg_percent
                )
                pos_scores = logits[(y_sub > 0.5).nonzero(as_tuple=True)[0]]
                neg_scores = logits[hard_negs]
                loss_push = pairwise_topk_push_loss(pos_scores, neg_scores, margin=rank_margin)

            weights = torch.ones_like(y_sub)
            if idx_prev is not None and lambda_prev != 0.0:
                weights = weights + lambda_prev * x_sub[:, -1, idx_prev].relu()

            loss_main = criterion(logits, y_sub.float(), weights=weights)

            reg = 0.0
            if hasattr(model, "gtg"):
                reg = model.gtg.entropy_reg(DEVICE)

            loss = loss_main + rank_weight * loss_push + reg
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            bcnt += 1

        logger.info(f"Epoch {e:03d} | loss={ep_loss / max(1, bcnt):.5f}")
    return model

# -------------------------------------------------------------------
# Utility functions for evaluation
# -------------------------------------------------------------------
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        if i < n_bins - 1:
            msk = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            msk = (probs >= bins[i]) & (probs <= bins[i + 1])
        if not np.any(msk):
            continue
        conf, acc = probs[msk].mean(), y_true[msk].mean()
        ece += (np.sum(msk) / n) * abs(acc - conf)
    return float(ece)


def np_threshold_at_fpr_strict(neg_scores: np.ndarray, target_fpr: float) -> float:
    if neg_scores.size == 0:
        return 0.5
    s = np.sort(neg_scores)
    k = max(1, int(math.ceil((1.0 - target_fpr) * s.size)))
    thr = s[k - 1]
    right = np.searchsorted(s, thr, side='right')
    if right == s.size:
        return float(np.nextafter(thr, np.inf))
    nxt = s[right]
    return float(np.nextafter(0.5 * (thr + nxt), np.inf))


def pot_threshold_gpd(
    neg_probs: np.ndarray,
    target_fpr: float,
    tail_quantile: float = 0.98,
    min_exceed: int = 500,
    min_frac: float = 0.005
) -> float:
    if neg_probs.size == 0:
        return 0.5
    q = np.quantile(neg_probs, tail_quantile)
    excess = neg_probs[neg_probs > q] - q
    if (excess.size < max(min_exceed, int(min_frac * max(1, neg_probs.size)))) or not _HAS_SCIPY:
        return np_threshold_at_fpr_strict(neg_probs, target_fpr)
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    p_tail = max(1e-12, 1.0 - tail_quantile)
    if abs(c) < 1e-8:
        y = -scale * math.log(target_fpr / p_tail)
    else:
        y = scale * ((target_fpr / p_tail) ** (-c) - 1.0) / c
    thr = float(q + max(0.0, y))
    return float(np.clip(thr, q, 1.0))


def conformal_fdr_daily(calib_scores, test_scores, calib_labels, test_dates, alpha_q=0.10):
    calib_null = calib_scores[calib_labels < 0.5]
    if calib_null.size == 0:
        return {}
    s = np.sort(calib_null)
    ranks = np.searchsorted(s, test_scores, side='right')
    pvals = (1.0 + ranks) / (s.size + 1.0)
    dates = pd.to_datetime(test_dates)
    res = {}
    df = pd.DataFrame({'p': pvals, 'date': dates})
    for d, g in df.groupby('date'):
        m = len(g)
        order = np.argsort(g['p'].values)
        p_sorted = g['p'].values[order]
        thresh = (np.arange(1, m + 1) / m) * alpha_q
        k = np.max(np.where(p_sorted <= thresh)[0]) + 1 if np.any(p_sorted <= thresh) else 0
        key = pd.Timestamp(d).date().isoformat()
        res[key] = {'n': int(m), 'k': int(k), 'rejected': int(k)}
    return res

# -------------------------------------------------------------------
# Extended metrics: high-wind, ranking, node-events
# -------------------------------------------------------------------
def build_node_events(y, nids, days):
    """
    Node-wise events: contiguous runs of label==1 in daily time.
    """
    y = np.asarray(y)
    nids = np.asarray(nids)
    days = np.asarray(days)
    idx = np.arange(len(y))

    order = np.lexsort((days.astype('datetime64[D]').astype('int64'), nids))
    y_s = y[order]
    n_s = nids[order]
    d_s = days[order]
    idx_s = idx[order]

    events = []
    start_j = None
    for j in range(len(y_s)):
        if y_s[j] >= 0.5:
            if start_j is None:
                start_j = j
            else:
                prev_j = j - 1
                if (n_s[j] != n_s[prev_j]) or (d_s[j] != d_s[prev_j] + np.timedelta64(1, 'D')):
                    s_j = start_j
                    e_j = prev_j
                    events.append({
                        'nid': int(n_s[s_j]),
                        'start_day': d_s[s_j],
                        'end_day': d_s[e_j],
                        'indices': idx_s[s_j:e_j + 1].copy()
                    })
                    start_j = j
        else:
            if start_j is not None:
                s_j = start_j
                e_j = j - 1
                events.append({
                    'nid': int(n_s[s_j]),
                    'start_day': d_s[s_j],
                    'end_day': d_s[e_j],
                    'indices': idx_s[s_j:e_j + 1].copy()
                })
                start_j = None

    if start_j is not None:
        s_j = start_j
        e_j = len(y_s) - 1
        events.append({
            'nid': int(n_s[s_j]),
            'start_day': d_s[s_j],
            'end_day': d_s[e_j],
            'indices': idx_s[s_j:e_j + 1].copy()
        })
    return events


def event_level_metrics(probs, y, nids, days, thr,
                        pre_window_days=0, post_window_days=0):
    probs = np.asarray(probs)
    y = np.asarray(y)
    nids = np.asarray(nids)
    days = np.asarray(days)

    events = build_node_events(y, nids, days)
    if not events:
        return dict(
            n_events=0,
            hits=0,
            misses=0,
            hit_rate=0.0,
            miss_rate=0.0,
            lead_time_mean=float('nan'),
            lead_time_std=float('nan'),
            lead_time_median=float('nan'),
            pre_window_days=int(pre_window_days),
            post_window_days=int(post_window_days)
        )

    hits = 0
    lead_times = []
    for ev in events:
        nid = ev['nid']
        s = ev['start_day'] - np.timedelta64(pre_window_days, 'D')
        e = ev['end_day'] + np.timedelta64(post_window_days, 'D')
        mask = (
            (nids == nid) &
            (days >= s) &
            (days <= e) &
            (probs >= thr)
        )
        if not mask.any():
            continue
        hits += 1
        detect_idx = np.where(mask)[0]
        detect_day = days[detect_idx].min()
        ld = (detect_day - ev['start_day']).astype('timedelta64[D]').astype(int)
        lead_times.append(int(ld))

    n_events = len(events)
    misses = n_events - hits
    if lead_times:
        lead_time_mean = float(np.mean(lead_times))
        lead_time_std = float(np.std(lead_times))
        lead_time_median = float(np.median(lead_times))
    else:
        lead_time_mean = lead_time_std = lead_time_median = float('nan')

    return dict(
        n_events=int(n_events),
        hits=int(hits),
        misses=int(misses),
        hit_rate=float(hits / max(1, n_events)),
        miss_rate=float(misses / max(1, n_events)),
        lead_time_mean=lead_time_mean,
        lead_time_std=lead_time_std,
        lead_time_median=lead_time_median,
        pre_window_days=int(pre_window_days),
        post_window_days=int(post_window_days)
    )


def high_wind_sample_metrics(probs, labels, wind, thr,
                             wind_threshold=None, wind_percentile=90):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    wind = np.asarray(wind)

    if wind_threshold is None:
        wind_threshold = float(np.percentile(wind, wind_percentile))

    mask = wind >= wind_threshold
    if not mask.any():
        return {
            "wind_threshold": float(wind_threshold),
            "n": 0,
            "pos_rate": 0.0,
            "roc_auc": float('nan'),
            "pr_auc": float('nan'),
            "cm": [[0, 0], [0, 0]],
            "acc": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "f1": 0.0
        }

    y = labels[mask]
    p = probs[mask]
    n = len(y)
    pos_rate = float(y.mean())

    if len(np.unique(y)) > 1:
        roc = float(roc_auc_score(y, p))
    else:
        roc = float('nan')
    pr = float(average_precision_score(y, p))

    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(cm[1, 1] / max(1, (cm[1, 1] + cm[1, 0])))

    return {
        "wind_threshold": float(wind_threshold),
        "n": int(n),
        "pos_rate": pos_rate,
        "roc_auc": roc,
        "pr_auc": pr,
        "cm": cm.tolist(),
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1
    }


def ranking_metrics_high_wind(probs, labels, days, wind,
                              wind_threshold=None, wind_percentile=90,
                              K_list=(5, 10, 20)):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    days = np.asarray(days)
    wind = np.asarray(wind)

    if wind_threshold is None:
        wind_threshold = float(np.percentile(wind, wind_percentile))

    mask = wind >= wind_threshold
    if not mask.any():
        return {
            "wind_threshold": float(wind_threshold),
            "precision_at_k": {int(k): 0.0 for k in K_list},
            "recall_at_k": {int(k): 0.0 for k in K_list},
            "n_days_used": 0
        }

    df = pd.DataFrame({
        "prob": probs[mask],
        "label": labels[mask],
        "day": days[mask].astype('datetime64[D]')
    })

    prec_numer = {int(k): 0.0 for k in K_list}
    prec_denom = {int(k): 0 for k in K_list}
    rec_numer = {int(k): 0.0 for k in K_list}
    rec_denom = {int(k): 0 for k in K_list}
    n_days_used = 0

    for d, g in df.groupby("day"):
        g_sorted = g.sort_values("prob", ascending=False)
        pos_total = int(g_sorted["label"].sum())
        if pos_total == 0:
            continue
        n_days_used += 1
        for K in K_list:
            k = int(K)
            topk = g_sorted.head(k)
            tp = int(topk["label"].sum())
            prec_numer[k] += tp
            prec_denom[k] += min(k, len(g_sorted))
            rec_numer[k] += tp
            rec_denom[k] += pos_total

    prec_at_k = {}
    rec_at_k = {}
    for K in K_list:
        k = int(K)
        prec_at_k[k] = float(prec_numer[k] / prec_denom[k]) if prec_denom[k] > 0 else 0.0
        rec_at_k[k] = float(rec_numer[k] / rec_denom[k]) if rec_denom[k] > 0 else 0.0

    return {
        "wind_threshold": float(wind_threshold),
        "precision_at_k": prec_at_k,
        "recall_at_k": rec_at_k,
        "n_days_used": int(n_days_used)
    }


def compute_extended_metrics(
    probs,
    y_true,
    nids,
    days,
    wind,
    thr_np,
    wind_percentile=90,
    K_list=(5, 10, 20),
    pre_window_days=0,
    post_window_days=0
):
    res = {}
    if wind is not None:
        res["high_wind_sample"] = high_wind_sample_metrics(
            probs, y_true, wind, thr_np,
            wind_threshold=None,
            wind_percentile=wind_percentile
        )
        res["ranking_high_wind"] = ranking_metrics_high_wind(
            probs, y_true, days, wind,
            wind_threshold=None,
            wind_percentile=wind_percentile,
            K_list=K_list
        )

    res["events_node_contig"] = event_level_metrics(
        probs, y_true, nids, days, thr_np,
        pre_window_days=pre_window_days,
        post_window_days=post_window_days
    )
    return res

# -------------------------------------------------------------------
# Forward collection (with optional wind feature)
# -------------------------------------------------------------------
@torch.no_grad()
def forward_collect(
    model,
    ds: SnapshotDataset,
    uniq_dates: np.ndarray,
    batch_size: int = 1,
    num_workers: int = 0,
    wind_idx: Optional[int] = None
):
    model.eval()
    model.to(DEVICE)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    all_logits = []
    all_y = []
    all_nids = []
    all_prox = []
    all_wind = []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _ = model(batch)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(batch.y.detach().cpu().numpy())
        all_nids.append(batch.nids.detach().cpu().numpy())
        all_prox.append(batch.prox.detach().cpu().numpy())
        if wind_idx is not None:
            w = batch.x[:, -1, wind_idx].detach().cpu().numpy()
            all_wind.append(w)

    all_dates_seq = []
    for t in range(len(ds)):
        n = len(ds.idx_lists[t])
        all_dates_seq.extend([pd.to_datetime(uniq_dates[t])] * n)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    nids = np.concatenate(all_nids, axis=0)
    prox = np.concatenate(all_prox, axis=0)
    all_dates = np.array(all_dates_seq, dtype='datetime64[D]')

    if wind_idx is not None and len(all_wind) > 0:
        wind_arr = np.concatenate(all_wind, axis=0)
    else:
        wind_arr = None

    return logits, y_true, nids, all_dates, prox, wind_arr

# -------------------------------------------------------------------
# Platt scaling
# -------------------------------------------------------------------
def platt_scale(logits: np.ndarray, labels: np.ndarray, max_iters: int = 2000, l2: float = 1e-3):
    s = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    opt = torch.optim.LBFGS([s, b], lr=0.5, max_iter=max_iters, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        a = torch.nn.functional.softplus(s) + 1e-3
        p = torch.sigmoid(a * x + b)
        loss = nn.functional.binary_cross_entropy(p, y) + l2 * (a * a + b * b)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr, flush=True)

    a = float(torch.nn.functional.softplus(s).item() + 1e-3)
    b_val = float(b.item())

    def transform(probs):
        z = np.asarray(probs, dtype=np.float64)
        l = _logit_np(z)
        return 1.0 / (1.0 + np.exp(-(a * l + b_val)))

    return "platt", transform, transform, transform

# -------------------------------------------------------------------
# Full evaluation with extended metrics
# -------------------------------------------------------------------
def evaluate_full(
    model,
    ds_train, tr_dates,
    ds_val, va_dates,
    ds_test, te_dates,
    grid_size=20,
    np_fpr_target=0.08,
    batch_size=1,
    num_workers=0,
    pool_days_for_np: int = 365,
    feature_names: Optional[List[str]] = None,
    wind_feat_name: str = 'wind_speed',
    wind_percentile: float = 90.0
):
    # --- identify wind feature index (optional) ---
    wind_idx = None
    if feature_names is not None:
        for i, nm in enumerate(feature_names):
            if nm.lower() == wind_feat_name.lower():
                wind_idx = i
                break
        if wind_idx is None:
            for i, nm in enumerate(feature_names):
                if wind_feat_name.lower() in nm.lower():
                    wind_idx = i
                    break

    # --- forward passes (raw logits) ---
    tr_logits, tr_y, tr_nids, tr_day, tr_prox, tr_wind = forward_collect(
        model, ds_train, tr_dates,
        batch_size=batch_size,
        num_workers=num_workers,
        wind_idx=wind_idx
    )
    va_logits, va_y, va_nids, va_day, va_prox, va_wind = forward_collect(
        model, ds_val, va_dates,
        batch_size=batch_size,
        num_workers=num_workers,
        wind_idx=wind_idx
    )
    te_logits, te_y, te_nids, te_day, te_prox, te_wind = forward_collect(
        model, ds_test, te_dates,
        batch_size=batch_size,
        num_workers=num_workers,
        wind_idx=wind_idx
    )

    # --- raw probs (before calibration) ---
    tr_probs_raw = sigmoid_np(tr_logits)
    va_probs_raw = sigmoid_np(va_logits)
    te_probs_raw = sigmoid_np(te_logits)

    # --- Platt calibration on logits, for calibration metrics only ---
    cal_name, trf_tr, trf_va, trf_te = platt_scale(tr_logits, tr_y)
    tr_probs_cal = trf_tr(tr_probs_raw)
    va_probs_cal = trf_va(va_probs_raw)
    te_probs_cal = trf_te(te_probs_raw)

    brier_pre = brier_score_loss(te_y, te_probs_raw)
    ece_pre = ece_score(te_y, te_probs_raw, 15)
    brier_post = brier_score_loss(te_y, te_probs_cal)
    ece_post = ece_score(te_y, te_probs_cal, 15)

    # ------------------------------------------------------------------
    # 1) NP-style threshold on NEGATIVE pool
    # ------------------------------------------------------------------
    if len(tr_day):
        tr_days_sorted = np.unique(tr_day)
        cutoff_date = tr_days_sorted[-1] - np.timedelta64(pool_days_for_np, 'D')
        pool_mask_train = tr_day >= cutoff_date
        pool_neg_train = tr_probs_raw[(tr_y < 0.5) & pool_mask_train]
    else:
        pool_neg_train = np.array([], dtype='float32')

    use_val = (va_y.sum() > 0)
    pool_neg_val = va_probs_raw[(va_y < 0.5)] if use_val else np.array([], dtype='float32')
    pooled_negs = np.concatenate([pool_neg_train, pool_neg_val], axis=0)

    if pooled_negs.size >= 100:
        thr_np = np_threshold_at_fpr_strict(pooled_negs, np_fpr_target)
        thr_pot_cand = pot_threshold_gpd(
            pooled_negs, np_fpr_target,
            tail_quantile=0.98, min_exceed=500, min_frac=0.005
        )
        fpr_est = (pooled_negs >= thr_pot_cand).mean() if pooled_negs.size > 0 else 1.0
        thr_pot = float(thr_pot_cand) if fpr_est <= np_fpr_target else float(thr_np)
    else:
        base_negs = va_probs_raw[va_y < 0.5] if (va_y < 0.5).any() else tr_probs_raw[tr_y < 0.5]
        thr_np = np_threshold_at_fpr_strict(base_negs, np_fpr_target)
        thr_pot = thr_np

    # ------------------------------------------------------------------
    # 2) validation-tuned F1 threshold
    # ------------------------------------------------------------------
    def find_best_f1_threshold(probs, labels, n_grid: int = 200):
        labels = np.asarray(labels)
        probs = np.asarray(probs)
        if labels.sum() < 5:
            return 0.5, 0.0, 0.0, 0.0

        qs = np.linspace(0.01, 0.99, n_grid)
        cand = np.quantile(probs, qs)
        best_f1 = -1.0
        best_thr = 0.5
        best_prec = 0.0
        best_rec = 0.0

        for th in cand:
            pred = (probs >= th).astype(int)
            if pred.sum() == 0:
                continue
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(th)
                best_prec = precision_score(labels, pred, zero_division=0)
                cm = confusion_matrix(labels, pred, labels=[0, 1])
                rec = cm[1, 1] / max(1, (cm[1, 1] + cm[1, 0]))
                best_rec = float(rec)

        if best_f1 < 0:
            pred = (probs >= 0.5).astype(int)
            best_f1 = f1_score(labels, pred, zero_division=0)
            best_prec = precision_score(labels, pred, zero_division=0)
            cm = confusion_matrix(labels, pred, labels=[0, 1])
            best_rec = cm[1, 1] / max(1, (cm[1, 1] + cm[1, 0]))
            best_thr = 0.5

        return best_thr, best_f1, best_prec, best_rec

    thr_f1_val, val_f1_at_thr, val_prec_at_thr, val_rec_at_thr = find_best_f1_threshold(
        va_probs_raw, va_y, n_grid=200
    )

    # ------------------------------------------------------------------
    # clamp thresholds into observed range
    # ------------------------------------------------------------------
    pmin, pmax = float(te_probs_raw.min()), float(te_probs_raw.max())

    def clamp_thr(th):
        if th >= pmax:
            return float(np.nextafter(pmax, -np.inf))
        if th <= pmin:
            return float(np.nextafter(pmin, np.inf))
        return float(th)

    thr_np = clamp_thr(thr_np)
    thr_pot = clamp_thr(thr_pot)
    thr_f1_val = clamp_thr(thr_f1_val)

    # ------------------------------------------------------------------
    # helpers to compute metrics
    # ------------------------------------------------------------------
    def point_metrics(probs, y_true, th):
        pred = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        tpr = (cm[1, 1] / max(1, (cm[1, 1] + cm[1, 0])))
        fpr = (cm[0, 1] / max(1, (cm[0, 1] + cm[0, 0])))
        prec = precision_score(y_true, pred, zero_division=0)
        return cm, acc, f1, tpr, fpr, prec

    def point_metrics_from_preds(y_true, preds):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        tpr = (cm[1, 1] / max(1, (cm[1, 1] + cm[1, 0])))
        fpr = (cm[0, 1] / max(1, (cm[0, 1] + cm[0, 0])))
        prec = precision_score(y_true, preds, zero_division=0)
        return cm, acc, f1, tpr, fpr, prec

    # ------------------------------------------------------------------
    # global ranking metrics
    # ------------------------------------------------------------------
    roc_auc = roc_auc_score(te_y, te_probs_cal) if len(np.unique(te_y)) > 1 else float('nan')
    pr_auc = average_precision_score(te_y, te_probs_cal)
    prec_at_k, rec_at_k = weekly_PR_at_K(te_probs_cal, te_y, te_day, K_list=(10, 20, 50, 100))

    # ------------------------------------------------------------------
    # test metrics at NP / POT / 0.5 / F1-val thresholds
    # ------------------------------------------------------------------
    cm_np, acc_np, f1_np, tpr_np, fpr_np, prec_np = point_metrics(te_probs_raw, te_y, thr_np)
    cm_pot, acc_pot, f1_pot, tpr_pot, fpr_pot, prec_pot = point_metrics(te_probs_raw, te_y, thr_pot)
    cm_05, acc_05, f1_05, tpr_05, fpr_05, prec_05 = point_metrics(te_probs_raw, te_y, 0.5)
    cm_f1, acc_f1, f1_f1, tpr_f1, fpr_f1, prec_f1 = point_metrics(te_probs_raw, te_y, thr_f1_val)

    # ------------------------------------------------------------------
    # post-processed (spatial NMS) predictions
    # ------------------------------------------------------------------
    pred_np_post = postprocess_3d_nms(te_probs_raw, te_nids, te_day, grid_size, thr_np)
    pred_pot_post = postprocess_3d_nms(te_probs_raw, te_nids, te_day, grid_size, thr_pot)
    pred_05_post = postprocess_3d_nms(te_probs_raw, te_nids, te_day, grid_size, 0.5)
    pred_f1_post = postprocess_3d_nms(te_probs_raw, te_nids, te_day, grid_size, thr_f1_val)

    cm_np_p, acc_np_p, f1_np_p, tpr_np_p, fpr_np_p, prec_np_p = point_metrics_from_preds(te_y, pred_np_post)
    cm_pot_p, acc_pot_p, f1_pot_p, tpr_pot_p, fpr_pot_p, prec_pot_p = point_metrics_from_preds(te_y, pred_pot_post)
    cm_05_p, acc_05_p, f1_05_p, tpr_05_p, fpr_05_p, prec_05_p = point_metrics_from_preds(te_y, pred_05_post)
    cm_f1_p, acc_f1_p, f1_f1_p, tpr_f1_p, fpr_f1_p, prec_f1_p = point_metrics_from_preds(te_y, pred_f1_post)

    # ------------------------------------------------------------------
    # diagnostics for negative pool
    # ------------------------------------------------------------------
    test_negs = te_probs_raw[te_y < 0.5]
    diag = {
        "pool_negs_size": int(pooled_negs.size),
        "pool_negs_q": (
            np.quantile(pooled_negs, [0.5, 0.9, 0.95, 0.99]).tolist()
            if pooled_negs.size > 0 else [float('nan')] * 4
        ),
        "test_negs_q": (
            np.quantile(test_negs, [0.5, 0.9, 0.95, 0.99]).tolist()
            if test_negs.size > 0 else [float('nan')] * 4
        )
    }

    # ------------------------------------------------------------------
    # conformal FDR per day (using calibrated probs)
    # ------------------------------------------------------------------
    conf_fdr = conformal_fdr_daily(
        calib_scores=va_probs_cal,
        test_scores=te_probs_cal,
        calib_labels=va_y,
        test_dates=te_day,
        alpha_q=0.10
    )

    # ------------------------------------------------------------------
    # extended metrics (NP and F1 thresholds) with a pre-window of 5 days
    # ------------------------------------------------------------------
    extended_np = compute_extended_metrics(
        probs=te_probs_raw,
        y_true=te_y,
        nids=te_nids,
        days=te_day,
        wind=te_wind,
        thr_np=thr_np,
        wind_percentile=wind_percentile,
        K_list=(5, 10, 20),
        pre_window_days=5,
        post_window_days=0
    )
    extended_f1 = compute_extended_metrics(
        probs=te_probs_raw,
        y_true=te_y,
        nids=te_nids,
        days=te_day,
        wind=te_wind,
        thr_np=thr_f1_val,
        wind_percentile=wind_percentile,
        K_list=(5, 10, 20),
        pre_window_days=5,
        post_window_days=0
    )

    return {
        "val": {
            "n": int(va_y.size),
            "pos_rate": float(va_y.mean()),
            "calibrator": cal_name,
            "thr_np_at_fpr": float(np_fpr_target),
            "thr_value": float(thr_np),
            "thr_pot_value": float(thr_pot),
            "thr_f1_val": float(thr_f1_val),
            "val_f1_at_thr_f1": float(val_f1_at_thr),
            "val_prec_at_thr_f1": float(val_prec_at_thr),
            "val_rec_at_thr_f1": float(val_rec_at_thr),
            "pooled_negs": int(pooled_negs.size)
        },
        "test_raw": {
            "n": int(te_y.size),
            "pos_rate": float(te_y.mean()),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "cm_np": cm_np.tolist(),
            "acc_np": float(acc_np),
            "prec_np": float(prec_np),
            "f1_np": float(f1_np),
            "tpr_np": float(tpr_np),
            "fpr_np": float(fpr_np),
            "cm_pot": cm_pot.tolist(),
            "acc_pot": float(acc_pot),
            "prec_pot": float(prec_pot),
            "f1_pot": float(f1_pot),
            "tpr_pot": float(tpr_pot),
            "fpr_pot": float(fpr_pot),
            "cm_05": cm_05.tolist(),
            "acc_05": float(acc_05),
            "prec_05": float(prec_05),
            "f1_05": float(f1_05),
            "tpr_05": float(tpr_05),
            "fpr_05": float(fpr_05),
            "cm_f1": cm_f1.tolist(),
            "acc_f1": float(acc_f1),
            "prec_f1": float(prec_f1),
            "f1_f1": float(f1_f1),
            "tpr_f1": float(tpr_f1),
            "fpr_f1": float(fpr_f1),
            "brier_pre": float(brier_pre),
            "ece_pre_15bins": float(ece_pre),
            "brier_post": float(brier_post),
            "ece_post_15bins": float(ece_post),
            "precision_at_k": {int(k): float(v) for k, v in prec_at_k.items()},
            "recall_at_k": {int(k): float(v) for k, v in rec_at_k.items()},
            "diagnostics": diag
        },
        "test_postproc": {
            "cm_np": cm_np_p.tolist(),
            "acc_np": float(acc_np_p),
            "prec_np": float(prec_np_p),
            "f1_np": float(f1_np_p),
            "tpr_np": float(tpr_np_p),
            "fpr_np": float(fpr_np_p),
            "cm_pot": cm_pot_p.tolist(),
            "acc_pot": float(acc_pot_p),
            "prec_pot": float(prec_pot_p),
            "f1_pot": float(f1_pot_p),
            "tpr_pot": float(tpr_pot_p),
            "fpr_pot": float(fpr_pot_p),
            "cm_05": cm_05_p.tolist(),
            "acc_05": float(acc_05_p),
            "prec_05": float(prec_05_p),
            "f1_05": float(f1_05_p),
            "tpr_05": float(tpr_05_p),
            "fpr_05": float(fpr_05_p),
            "cm_f1": cm_f1_p.tolist(),
            "acc_f1": float(acc_f1_p),
            "prec_f1": float(prec_f1_p),
            "f1_f1": float(f1_f1_p),
            "tpr_f1": float(tpr_f1_p),
            "fpr_f1": float(fpr_f1_p)
        },
        "conformal_fdr_daily": conf_fdr,
        "extended": {
            "np": extended_np,
            "val_f1": extended_f1
        }
    }

# -------------------------------------------------------------------
# Explainability: gradient×input per (lag, feature) for TP nodes
# -------------------------------------------------------------------
@torch.enable_grad()
def explain_true_positives_tgcn(
    model: nn.Module,
    ds: SnapshotDataset,
    probs: np.ndarray,
    y_true: np.ndarray,
    nids_arr: np.ndarray,
    dates_arr: np.ndarray,
    thr: float,
    feature_names: List[str],
    max_examples: int = 10,
    topk: int = 5
):
    explain_device = torch.device("cpu")
    orig_device = next(model.parameters()).device
    was_training = model.training

    model.eval()
    model.to(explain_device)

    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    nids_arr = np.asarray(nids_arr, dtype=np.int64)
    dates_arr = np.asarray(dates_arr, dtype='datetime64[D]')

    pred = (probs >= thr).astype(int)
    tp_idx = np.where((pred == 1) & (y_true > 0.5))[0]
    if tp_idx.size == 0:
        model.to(orig_device)
        if was_training:
            model.train()
        return []

    spans = []
    c = 0
    for k in range(len(ds.idx_lists)):
        N = len(ds.idx_lists[k])
        spans.append((c, c + N))
        c += N

    out = []

    for ridx in tp_idx[np.argsort(-probs[tp_idx])][:max_examples]:
        snap = 0
        for k, (s, e) in enumerate(spans):
            if s <= ridx < e:
                snap = k
                break
        local = ridx - spans[snap][0]

        d: Data = ds[snap]

        x = d.x.clone().to(explain_device).detach()
        x.requires_grad_(True)

        data = Data(
            x=x,
            edge_index=d.edge_index.clone().to(explain_device),
            y=d.y.clone().to(explain_device),
            nids=d.nids.clone().to(explain_device),
            prox=d.prox.clone().to(explain_device) if hasattr(d, "prox") else None,
        )

        logits, _ = model(data)
        node_idx = int(local)
        prob_t = torch.sigmoid(logits[node_idx])

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        prob_t.backward()

        g_node = x.grad[node_idx].detach().cpu().numpy()   # [T,F]
        v_node = x[node_idx].detach().cpu().numpy()        # [T,F]

        contrib = g_node * v_node
        imp = np.abs(contrib)
        T_steps, F = imp.shape

        flat_idx = np.argsort(-imp.reshape(-1))[:topk]
        reasons = []
        for fi in flat_idx:
            t_idx = fi // F
            f_idx = fi % F
            lag = (T_steps - 1) - t_idx
            reasons.append(
                dict(
                    feature=feature_names[f_idx],
                    lag_steps=int(lag),
                    value=float(v_node[t_idx, f_idx]),
                    importance=float(imp[t_idx, f_idx]),
                )
            )

        out.append(
            dict(
                node=int(d.nids[local].item()),
                date=str(pd.to_datetime(dates_arr[ridx]).date()),
                prob=float(probs[ridx]),
                reasons=reasons,
            )
        )

    model.to(orig_device)
    if was_training:
        model.train()

    return out

# -------------------------------------------------------------------
# Data building helper for ablations
# -------------------------------------------------------------------
def build_data_and_datasets(
    grid_size: int = 20,
    lookback_steps: int = 180,
    csv_path: str = 'GRID3km_DAILY_20x20_modelready_labeled_featured.csv',
    out_dir: str = 'cache_daily_v6'
):
    GRID = grid_size
    LOOKBACK_STEPS = lookback_steps
    CSV = csv_path
    OUT = out_dir

    logger.info(f"Loading daily CSV: {CSV}")
    df_daily = load_daily(CSV, grid_size=GRID)
    df_daily = impute_missing_per_node(df_daily)
    df_daily = add_static_priors(df_daily, grid_size=GRID)

    df_daily['date'] = pd.to_datetime(df_daily['date']).dt.normalize()
    dates_sorted = np.sort(df_daily['date'].unique())
    test_frac = 0.2
    test_cut = dates_sorted[int((1 - test_frac) * len(dates_sorted))]

    train_full = df_daily[df_daily['date'] <= test_cut].copy()
    test_df = df_daily[df_daily['date'] > test_cut].copy()

    tr_dates_sorted = np.sort(train_full['date'].unique())
    val_cut = tr_dates_sorted[int(0.9 * len(tr_dates_sorted))]
    val_df = train_full[train_full['date'] > val_cut].copy()
    train_df = train_full[train_full['date'] <= val_cut].copy()

    ignore_cols = {'node', 'date', 'label', 'node_id', 'visi_proximity', 'visi_proximity_inv'}
    feature_cols = [c for c in df_daily.columns
                    if (df_daily[c].dtype.kind in 'fc') and (c not in ignore_cols)]
    prox_col = 'visi_proximity_inv'

    train_df = ensure_node_and_date(train_df, GRID)
    val_df = ensure_node_and_date(val_df, GRID)
    test_df = ensure_node_and_date(test_df, GRID)

    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df, feature_cols)

    tr_seq, tr_lbl, tr_nid, tr_dat, tr_prox, TR_WIN = create_windows(
        train_df, feature_cols, prox_col, LOOKBACK_STEPS, OUT, '_train',
        min_window=32, drop_short_nodes=True
    )
    va_seq, va_lbl, va_nid, va_dat, va_prox, VA_WIN = create_windows(
        val_df, feature_cols, prox_col, LOOKBACK_STEPS, OUT, '_val',
        min_window=32, drop_short_nodes=False
    )
    te_seq, te_lbl, te_nid, te_dat, te_prox, TE_WIN = create_windows(
        test_df, feature_cols, prox_col, LOOKBACK_STEPS, OUT, '_test',
        min_window=32, drop_short_nodes=False
    )

    EFF_WIN = int(min(TR_WIN, VA_WIN, TE_WIN))
    in_channels = int(tr_seq.shape[-1])

    del df_daily, train_full, train_df, val_df, test_df
    gc.collect()

    tr_dates, tr_idx, edge_index = build_snapshot_index(tr_dat, grid_size=GRID)
    va_dates, va_idx, _ = build_snapshot_index(va_dat, grid_size=GRID)
    te_dates, te_idx, _ = build_snapshot_index(te_dat, grid_size=GRID)

    train_ds = SnapshotDataset(tr_seq, tr_lbl, tr_nid, tr_prox, tr_dates, tr_idx, edge_index)
    val_ds = SnapshotDataset(va_seq, va_lbl, va_nid, va_prox, va_dates, va_idx, edge_index)
    test_ds = SnapshotDataset(te_seq, te_lbl, te_nid, te_prox, te_dates, te_idx, edge_index)

    train_ds.feature_names = feature_cols

    return (GRID, EFF_WIN, in_channels,
            feature_cols,
            train_ds, val_ds, test_ds,
            tr_lbl, tr_dates, va_dates, te_dates,
            edge_index, scaler)

# -------------------------------------------------------------------
# Model builder
# -------------------------------------------------------------------
def build_model(model_type: str, in_channels: int, hidden: int, dropout: float,
                lookback_steps: int, edge_index: torch.Tensor):
    if model_type == "tgcn_full":
        return TGCNContrast(in_channels=in_channels, hidden=hidden,
                            dropout=dropout, lookback_steps=lookback_steps,
                            use_gtg=True)
    elif model_type == "tgcn_no_gtg":
        return TGCNContrast(in_channels=in_channels, hidden=hidden,
                            dropout=dropout, lookback_steps=lookback_steps,
                            use_gtg=False)
    elif model_type == "gru_only":
        return GRUOnly(in_channels=in_channels, hidden=hidden,
                       dropout=dropout, lookback_steps=lookback_steps)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# -------------------------------------------------------------------
# Ablation main: run multiple 10-epoch experiments
# -------------------------------------------------------------------
def ablation_main():
    (GRID, EFF_WIN, in_channels,
     feature_cols,
     train_ds, val_ds, test_ds,
     tr_lbl, tr_dates, va_dates, te_dates,
     edge_index, scaler) = build_data_and_datasets()

    prev = float((tr_lbl > 0.5).sum() / max(1, len(tr_lbl)))
    if prev > 0 and prev < 0.5:
        pos_weight = math.sqrt((1.0 - prev) / prev)
    else:
        pos_weight = 1.0
    logger.info(f"Global pos_rate={prev:.6f}, pos_weight(BCE)≈{pos_weight:.3f}")

    experiments = [
        dict(name="tgcn_bce_rank0_tp01",  model_type="tgcn_full",   loss_type="bce",   gamma=1.0, alpha=None,  rank_weight=0.0,  target_pos_ratio=0.1),
        dict(name="tgcn_bce_rank02_tp01", model_type="tgcn_full",   loss_type="bce",   gamma=1.0, alpha=None,  rank_weight=0.2,  target_pos_ratio=0.1),
        dict(name="tgcn_bce_rank02_tp03", model_type="tgcn_full",   loss_type="bce",   gamma=1.0, alpha=None,  rank_weight=0.2,  target_pos_ratio=0.3),
        dict(name="tgcn_focal_g1_a025",   model_type="tgcn_full",   loss_type="focal", gamma=1.0, alpha=0.25,  rank_weight=0.2,  target_pos_ratio=0.1),
        dict(name="tgcn_focal_g2_a025",   model_type="tgcn_full",   loss_type="focal", gamma=2.0, alpha=0.25,  rank_weight=0.2,  target_pos_ratio=0.1),
        dict(name="tgcn_focal_g1_a05",    model_type="tgcn_full",   loss_type="focal", gamma=1.0, alpha=0.5,   rank_weight=0.2,  target_pos_ratio=0.1),
        dict(name="tgcn_noGTG_bce",       model_type="tgcn_no_gtg", loss_type="bce",   gamma=1.0, alpha=None,  rank_weight=0.0,  target_pos_ratio=0.1),
        dict(name="tgcn_noGTG_focal",     model_type="tgcn_no_gtg", loss_type="focal", gamma=1.0, alpha=0.25,  rank_weight=0.2,  target_pos_ratio=0.1),
        dict(name="gru_bce",              model_type="gru_only",    loss_type="bce",   gamma=1.0, alpha=None,  rank_weight=0.0,  target_pos_ratio=0.1),
        dict(name="gru_focal",            model_type="gru_only",    loss_type="focal", gamma=1.0, alpha=0.25,  rank_weight=0.0,  target_pos_ratio=0.1),
    ]

    os.makedirs("ablation_results", exist_ok=True)

    summary = {}

    for i, cfg in enumerate(experiments):
        logger.info("=" * 80)
        logger.info(f"Experiment {i+1}/{len(experiments)}: {cfg['name']}")
        logger.info("=" * 80)

        random.seed(SEED + i)
        np.random.seed(SEED + i)
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed_all(SEED + i)

        model = build_model(
            model_type=cfg["model_type"],
            in_channels=in_channels,
            hidden=256,
            dropout=0.3,
            lookback_steps=EFF_WIN,
            edge_index=edge_index,
        )

        if cfg["loss_type"] == "focal":
            f_alpha = cfg.get("alpha", None)
            pw = None
        else:
            f_alpha = None
            pw = pos_weight

        model = train(
            model,
            train_ds,
            epochs=10,
            lr=7e-4,
            batch_size=1,
            grid_size=GRID,
            loss_type=cfg["loss_type"],
            f_alpha=f_alpha,
            gamma=cfg.get("gamma", 3.0),
            rank_margin=0.3,
            rank_weight=cfg["rank_weight"],
            target_pos_ratio=cfg["target_pos_ratio"],
            num_workers=0,
            top_neg_percent=0.02,
            lambda_prev=0.3,
            feat_names=feature_cols,
            pos_weight=pw,
        )

        res = evaluate_full(
            model,
            train_ds, tr_dates,
            val_ds, va_dates,
            test_ds, te_dates,
            grid_size=GRID,
            np_fpr_target=0.08,
            batch_size=1,
            num_workers=0,
            pool_days_for_np=365,
            feature_names=feature_cols,
            wind_feat_name='wind_speed',
            wind_percentile=90.0
        )

        # compact metrics for quick comparison
        sr = res["test_raw"]
        summary[cfg["name"]] = {
            "config": cfg,
            "val_pos_rate": res["val"]["pos_rate"],
            "val_thr_f1": res["val"]["thr_f1_val"],
            "val_f1_at_thr_f1": res["val"]["val_f1_at_thr_f1"],
            "test_pos_rate": sr["pos_rate"],
            "roc_auc": sr["roc_auc"],
            "pr_auc": sr["pr_auc"],
            "f1_np": sr["f1_np"],
            "f1_05": sr["f1_05"],
            "f1_f1": sr["f1_f1"],
            "precision_at_k": sr["precision_at_k"],
            "recall_at_k": sr["recall_at_k"],
        }

        # save full result per experiment
        with open(os.path.join("ablation_results", f"{cfg['name']}_metrics.json"), "w") as f:
            json.dump(res, f, indent=2)

        # save model weights
        torch.save(model.state_dict(), os.path.join("ablation_results", f"{cfg['name']}_model.pt"))

    # save global summary
    with open("ablation_results_daily_v6.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Ablation finished. Summary written to ablation_results_daily_v6.json")

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    ablation_main()
