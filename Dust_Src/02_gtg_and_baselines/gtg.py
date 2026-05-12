#!/usr/bin/env python3
# Space–Time Graph pipeline (GTG-TGCN) with OOM fixes:
#  1) Actually restrict the dataset to the GTG horizon BEFORE building x/edges
#  2) Remove float32 duplication in softmax (no .float() on huge tensors)
#  3) Chunk GTG history/attention computation so peak memory is bounded
#  4) Fix explanations to compute attention ONLY for selected nodes (no full-N return_att)

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ---------------------------------------------------------------------------
# Data loading and graph construction
# ---------------------------------------------------------------------------

def load_daily_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"node", "date", "label"}.issubset(df.columns), "CSV must have node,date,label"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "node"]).reset_index(drop=True)
    print(f"Loading CSV: {path}")
    print(f"Rows: {len(df)}, columns: {df.shape[1]}")
    return df


def encode_grid_nodes(df: pd.DataFrame):
    unique_nodes = sorted(df["node"].unique())
    rows = []
    cols = []
    for s in unique_nodes:
        r, c = s.split("_")
        rows.append(int(r))
        cols.append(int(c))
    rows = np.array(rows)
    cols = np.array(cols)

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    grid_h = rmax - rmin + 1
    grid_w = cmax - cmin + 1

    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    idx_to_rc = []
    for n in unique_nodes:
        r, c = n.split("_")
        idx_to_rc.append((int(r) - rmin, int(c) - cmin))

    return node_to_idx, idx_to_rc, grid_h, grid_w


def normalize_features(X: np.ndarray):
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    Xn = (X - mean) / std
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=10.0, neginf=-10.0)
    return Xn, mean, std


def angle_to_direction(sin_val: float, cos_val: float):
    angle = math.atan2(sin_val, cos_val)  # [-pi, pi], 0 ~ East
    if angle < 0:
        angle += 2 * math.pi
    sector = int((angle / (2 * math.pi)) * 8) % 8

    directions = {
        0: (0, 1),    # East
        1: (-1, 1),   # NE
        2: (-1, 0),   # North
        3: (-1, -1),  # NW
        4: (0, -1),   # West
        5: (1, -1),   # SW
        6: (1, 0),    # South
        7: (1, 1),    # SE
    }
    return sector, directions[sector]


def build_space_time_graph(
    df: pd.DataFrame,
    use_wind_edges: bool = True,
    wind_feature_name: str = "wind_speed",
    # NEW: hard cut to GTG horizon BEFORE building x/edges
    restrict_to_gtg_horizon: bool = True,
    gtg_years: float = 12.0,
    total_years: float = 20.0,
) -> Data:
    node_to_idx, idx_to_rc, grid_h, grid_w = encode_grid_nodes(df)

    exclude_cols = {"node", "date", "label"}
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    feat_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    print(f"Using {len(feat_cols)} feature columns.")
    if wind_feature_name not in df.columns:
        print(f"WARNING: wind_feature_name='{wind_feature_name}' not in CSV; high-wind metrics disabled.")
    else:
        print(f"Using '{wind_feature_name}' as wind feature for high-wind percentiles.")

    # Dates and optional GTG horizon restriction
    unique_dates_all = np.sort(df["date"].unique())
    N_dates_all = len(unique_dates_all)

    if restrict_to_gtg_horizon:
        gtg_frac = gtg_years / total_years
        gtg_Tmax = int(gtg_frac * N_dates_all)
        keep_dates = set(unique_dates_all[:gtg_Tmax])
        df = df[df["date"].isin(keep_dates)].copy()
        unique_dates = np.sort(df["date"].unique())
        print(f"Restricting graph to GTG horizon: {len(unique_dates)}/{N_dates_all} dates")
    else:
        unique_dates = unique_dates_all
        gtg_Tmax = len(unique_dates)

    date_to_tid = {d: i for i, d in enumerate(unique_dates)}
    T = len(unique_dates)

    # Build mapping (local_node_idx, t) -> node-id
    nt_to_nid = {}
    nid_to_nt = []
    X_rows = []
    Y_rows = []
    wind_rows = []

    for _, row in df.iterrows():
        n_str = row["node"]
        d = row["date"]
        t = date_to_tid[d]
        local_node_idx = node_to_idx[n_str]
        key = (local_node_idx, t)
        if key in nt_to_nid:
            continue
        nid = len(nt_to_nid)
        nt_to_nid[key] = nid
        nid_to_nt.append(key)
        X_rows.append(row[feat_cols].values.astype(np.float32))
        Y_rows.append(float(row["label"]))
        if wind_feature_name in df.columns:
            wind_rows.append(float(row[wind_feature_name]))

    X = np.stack(X_rows, axis=0)  # raw features
    y = np.array(Y_rows, dtype=np.float32)
    print(f"Built {len(nt_to_nid)} nodes across {T} time steps.")

    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

    if wind_rows:
        wind_vals = np.array(wind_rows, dtype=np.float32)
        wind_vals = np.nan_to_num(wind_vals, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        wind_vals = np.zeros(len(nid_to_nt), dtype=np.float32)

    # ------------------------------------------------------------------
    # Time split inside GTG horizon (BEFORE normalization)
    # ------------------------------------------------------------------
    train_T = int(0.7 * T)
    val_T   = int(0.85 * T)

    train_mask = torch.zeros(X.shape[0], dtype=torch.bool)
    val_mask   = torch.zeros(X.shape[0], dtype=torch.bool)
    test_mask  = torch.zeros(X.shape[0], dtype=torch.bool)

    for nid, (_, t) in enumerate(nid_to_nt):
        if t < train_T:
            train_mask[nid] = True
        elif t < val_T:
            val_mask[nid] = True
        else:
            test_mask[nid] = True

    # Boolean numpy masks for indexing X
    train_idx = train_mask.numpy()
    val_idx   = val_mask.numpy()
    test_idx  = test_mask.numpy()

    # ------------------------------------------------------------------
    # Normalize features AFTER splitting, each split on itself
    # ------------------------------------------------------------------
    X_norm = np.zeros_like(X, dtype=np.float32)

    if train_idx.any():
        X_train_norm, train_mean, train_std = normalize_features(X[train_idx])
        X_norm[train_idx] = X_train_norm
    else:
        train_mean = train_std = None

    if val_idx.any():
        X_val_norm, val_mean, val_std = normalize_features(X[val_idx])
        X_norm[val_idx] = X_val_norm
    else:
        val_mean = val_std = None

    if test_idx.any():
        X_test_norm, test_mean, test_std = normalize_features(X[test_idx])
        X_norm[test_idx] = X_test_norm
    else:
        test_mean = test_std = None

    # ------------------------------------------------------------------
    # Graph edges (use X_norm for wind_dir-based temporal edges)
    # ------------------------------------------------------------------
    # Speedups: rc -> local index lookup
    rc_to_local = {(r, c): li for li, (r, c) in enumerate(idx_to_rc)}

    # Build t -> nids
    t_to_nids = defaultdict(list)
    for nid, (_, t) in enumerate(nid_to_nt):
        t_to_nids[t].append(nid)

    # Spatial edges (4-neighbour per time slice)
    edge_spat_src = []
    edge_spat_dst = []
    neighbours_by_local = {}
    for li, (r, c) in enumerate(idx_to_rc):
        nb = []
        for (nr, nc) in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                li2 = rc_to_local.get((nr, nc), None)
                if li2 is not None:
                    nb.append(li2)
        neighbours_by_local[li] = nb

    for t in range(T):
        for nid in t_to_nids[t]:
            local_idx, _ = nid_to_nt[nid]
            for nb_local in neighbours_by_local[local_idx]:
                key2 = (nb_local, t)
                nid2 = nt_to_nid.get(key2, None)
                if nid2 is not None:
                    edge_spat_src.append(nid)
                    edge_spat_dst.append(nid2)
                    edge_spat_src.append(nid2)
                    edge_spat_dst.append(nid)

    # Temporal edges
    edge_temp_src = []
    edge_temp_dst = []

    try:
        sin_idx = feat_cols.index("wind_dir_sin")
        cos_idx = feat_cols.index("wind_dir_cos")
        has_wind_dir = True
    except ValueError:
        has_wind_dir = False
        print("WARNING: wind_dir_sin / wind_dir_cos not found; temporal edges self-only.")

    for t in range(1, T):
        for nid_t in t_to_nids[t]:
            local_idx, _ = nid_to_nt[nid_t]
            r, c = idx_to_rc[local_idx]
            prev_key = (local_idx, t - 1)
            prev_nid = nt_to_nid.get(prev_key, None)
            if prev_nid is None:
                continue

            dst_nid_main = nid_t

            if use_wind_edges and has_wind_dir:
                prev_feat = X_norm[prev_nid]
                sin_val = prev_feat[sin_idx]
                cos_val = prev_feat[cos_idx]
                _, (dr, dc) = angle_to_direction(sin_val, cos_val)
                nr, nc = r + dr, c + dc

                neighbour_local_idx = rc_to_local.get((nr, nc), None)
                if neighbour_local_idx is not None:
                    dst_key = (neighbour_local_idx, t)
                    dst_nid = nt_to_nid.get(dst_key, None)
                    if dst_nid is not None:
                        dst_nid_main = dst_nid

                edge_temp_src.append(prev_nid)
                edge_temp_dst.append(dst_nid_main)
                if dst_nid_main != nid_t:
                    edge_temp_src.append(prev_nid)
                    edge_temp_dst.append(nid_t)
            else:
                edge_temp_src.append(prev_nid)
                edge_temp_dst.append(nid_t)

    x = torch.from_numpy(X_norm).float()
    y_t = torch.from_numpy(y).float()
    wind_t = torch.from_numpy(wind_vals).float()

    edge_index_spat = torch.tensor([edge_spat_src, edge_spat_dst], dtype=torch.long)
    edge_index_temp = torch.tensor([edge_temp_src, edge_temp_dst], dtype=torch.long)

    # Node/time indexing for metrics
    n_nodes = len(node_to_idx)
    T_steps = T

    nid_node_idx = np.zeros(len(nid_to_nt), dtype=np.int64)
    nid_time_idx = np.zeros(len(nid_to_nt), dtype=np.int64)
    for nid, (local_idx, t) in enumerate(nid_to_nt):
        nid_node_idx[nid] = local_idx
        nid_time_idx[nid] = t

    node_t_to_nid = -np.ones((n_nodes, T_steps), dtype=np.int64)
    for nid, (local_idx, t) in enumerate(nid_to_nt):
        node_t_to_nid[local_idx, t] = nid

    data = Data(
        x=x,
        y=y_t,
        edge_index_spat=edge_index_spat,
        edge_index_temp=edge_index_temp,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        wind=wind_t,
    )
    data.n_nodes = n_nodes
    data.T = T_steps
    data.local_node_idx = torch.from_numpy(nid_node_idx).long()
    data.time_idx = torch.from_numpy(nid_time_idx).long()
    data.node_t_to_nid = torch.from_numpy(node_t_to_nid).long()
    data.feat_cols = feat_cols
    data.dates = [str(d) for d in unique_dates]
    data.gtg_Tmax = T  # after restriction, GTG horizon == full graph

    # (optional) keep stats if useful
    data.train_mean = None if train_mean is None else torch.from_numpy(train_mean.squeeze(0)).float()
    data.train_std  = None if train_std  is None else torch.from_numpy(train_std.squeeze(0)).float()
    data.val_mean   = None if val_mean   is None else torch.from_numpy(val_mean.squeeze(0)).float()
    data.val_std    = None if val_std    is None else torch.from_numpy(val_std.squeeze(0)).float()
    data.test_mean  = None if test_mean  is None else torch.from_numpy(test_mean.squeeze(0)).float()
    data.test_std   = None if test_std   is None else torch.from_numpy(test_std.squeeze(0)).float()

    return data


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class EarlyWarningStats:
    events: int
    hit_rate: float
    miss_rate: float
    mean_lead: float
    median_lead: float
    window_days: int
    thr: float
    fa_rate: float
    fa_count: int
    fa_per_event: float


@dataclass
class LowFAThresholdResult:
    thr: float
    hit_rate: float
    miss_rate: float
    fa_rate: float
    fa_per_event: float
    cost: float


class SAGEClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = self.dropout(h)
        return self.out(h).squeeze(-1)


class GDNStyleClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.3, recon_weight=0.1):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden_dim, 1)
        self.recon_head = nn.Linear(hidden_dim, in_dim)
        self.recon_weight = recon_weight

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = self.dropout(h)
        logits = self.cls_head(h).squeeze(-1)
        recon = self.recon_head(h)
        return logits, recon

    def loss(self, logits, recon, x, y, pos_weight=None):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=10.0, neginf=-10.0)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, y) if pos_weight is not None else nn.BCEWithLogitsLoss()(logits, y)
        recon_loss = ((recon - x) ** 2).mean()
        return bce + self.recon_weight * recon_loss


class GraphWaveNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.gate_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        h_conv = self.conv(x, edge_index)
        gate = torch.sigmoid(self.gate_lin(x))
        return torch.tanh(h_conv) * gate


class GraphWaveNetClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphWaveNetLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.dropout(h)
        return self.out(h).squeeze(-1)


class SimpleGATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, heads=2, num_layers=2, dropout=0.3):
        super().__init__()
        gat_hidden = hidden_dim // heads
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, gat_hidden, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, gat_hidden, heads=heads, concat=True))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = self.dropout(h)
        return self.out(h).squeeze(-1)


class GTG_TGCN_Classifier(nn.Module):
    """
    OOM-safe GTG:
      - Computes fast/med/slow group scalars in chunks (no [N,F,L] full tensors)
      - Avoids float32 duplication in softmax
      - Explanations computed only for selected node indices
    """

    def __init__(
        self,
        in_dim,
        fast_idx,
        med_idx,
        slow_idx,
        node_t_to_nid: torch.Tensor,
        local_node_idx: torch.Tensor,
        time_idx: torch.Tensor,
        hidden_dim=32,
        num_layers=3,
        dropout=0.3,
        fast_lags: int = 14,
        med_lags: int = 112,
        slow_lags: int = 36,
        med_stride: int = 7,
        slow_stride: int = 30,
        gtg_chunk: int = 50_000,  # NEW: chunk size
    ):
        super().__init__()
        self.fast_idx = torch.tensor(fast_idx, dtype=torch.long)
        self.med_idx = torch.tensor(med_idx, dtype=torch.long)
        self.slow_idx = torch.tensor(slow_idx, dtype=torch.long)

        F_fast = len(fast_idx)
        F_med = len(med_idx)
        F_slow = len(slow_idx)

        self.register_buffer("node_t_to_nid", node_t_to_nid.long())
        self.register_buffer("local_node_idx", local_node_idx.long())
        self.register_buffer("time_idx", time_idx.long())

        self.register_buffer("fast_lag_offsets", torch.arange(fast_lags, dtype=torch.long))
        self.register_buffer("med_lag_offsets", torch.arange(med_lags, dtype=torch.long) * med_stride)
        self.register_buffer("slow_lag_offsets", torch.arange(slow_lags, dtype=torch.long) * slow_stride)

        self.w_fast = nn.Parameter(torch.randn(F_fast, fast_lags))
        self.w_med = nn.Parameter(torch.randn(F_med, med_lags))
        self.w_slow = nn.Parameter(torch.randn(F_slow, slow_lags))

        self.lin_gtg = nn.Linear(3, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

        self.gtg_chunk = int(gtg_chunk)

    def _hist_dtype(self, x_all: torch.Tensor):
        if x_all.is_cuda:
            return torch.float16
        return x_all.dtype

    def _build_group_history_slice(
        self,
        x_all: torch.Tensor,
        group_idx: torch.Tensor,
        lag_offsets: torch.Tensor,
        idx_slice: torch.Tensor,  # 1D tensor of node indices
    ) -> torch.Tensor:
        device = x_all.device
        idx_slice = idx_slice.to(device)

        local_nodes = self.local_node_idx[idx_slice].to(device)
        time_idx = self.time_idx[idx_slice].to(device)
        node_t_to_nid = self.node_t_to_nid.to(device)

        B = local_nodes.numel()
        Fg = group_idx.numel()
        L = lag_offsets.numel()

        hist_dtype = self._hist_dtype(x_all)
        hist = torch.zeros((B, Fg, L), device=device, dtype=hist_dtype)
        if Fg == 0 or L == 0 or B == 0:
            return hist

        group_idx = group_idx.to(device)
        group_feats_now = x_all[idx_slice][:, group_idx].to(dtype=hist_dtype)

        for ell in range(L):
            lag = int(lag_offsets[ell].item())
            if lag == 0:
                hist[:, :, ell] = group_feats_now
                continue

            target_t = time_idx - lag
            valid = target_t >= 0
            if not valid.any():
                continue

            src_nodes = local_nodes[valid]
            src_times = target_t[valid]
            hist_nids = node_t_to_nid[src_nodes, src_times]
            valid2 = hist_nids >= 0
            if not valid2.any():
                continue

            feats = x_all[hist_nids[valid2]][:, group_idx].to(dtype=hist_dtype)
            dest_idx = torch.where(valid)[0][valid2]
            hist[dest_idx, :, ell] = feats

        return hist

    @staticmethod
    def _stable_softmax_1d(s_flat: torch.Tensor) -> torch.Tensor:
        # subtract max to improve stability; stays in fp16/bf16
        s_flat = s_flat - s_flat.max(dim=1, keepdim=True).values
        return torch.softmax(s_flat, dim=1)

    def _group_scalar_chunked(
        self,
        x_all: torch.Tensor,
        group_idx: torch.Tensor,
        lag_offsets: torch.Tensor,
        w_param: torch.Tensor,
    ) -> torch.Tensor:
        device = x_all.device
        N = x_all.size(0)
        if group_idx.numel() == 0 or lag_offsets.numel() == 0:
            return x_all.new_zeros((N, 1))

        group_idx = group_idx.to(device)
        lag_offsets = lag_offsets.to(device)

        outs = []
        chunk = self.gtg_chunk

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idx = torch.arange(start, end, device=device, dtype=torch.long)

            hist = self._build_group_history_slice(x_all, group_idx, lag_offsets, idx)  # [B,F,L]
            if hist.numel() == 0:
                outs.append(x_all.new_zeros((end - start, 1)))
                continue

            w = w_param.to(device=device, dtype=hist.dtype).unsqueeze(0)  # [1,F,L]
            s = hist * w
            s_flat = s.view(s.size(0), -1)  # [B,F*L]
            att_flat = self._stable_softmax_1d(s_flat).view_as(s)  # [B,F,L]
            h = (att_flat * hist).sum(dim=(1, 2), dtype=torch.float32).unsqueeze(-1)  # [B,1]
            outs.append(h.to(dtype=x_all.dtype))

        return torch.cat(outs, dim=0)

    def forward(self, x, edge_index_spat, edge_index_temp):
        device = x.device

        # GTG scalars (always produce [N,1] each)
        h_fast = self._group_scalar_chunked(x, self.fast_idx, self.fast_lag_offsets, self.w_fast)
        h_med  = self._group_scalar_chunked(x, self.med_idx,  self.med_lag_offsets,  self.w_med)
        h_slow = self._group_scalar_chunked(x, self.slow_idx, self.slow_lag_offsets, self.w_slow)

        # [N,3] -> make sure dtype matches lin_gtg weights (avoids Half/Float issues)
        h_gtg = torch.cat([h_fast, h_med, h_slow], dim=1)
        h_gtg = h_gtg.to(dtype=self.lin_gtg.weight.dtype)

        # DEFINE h unconditionally
        h = torch.relu(self.lin_gtg(h_gtg))

        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = self.dropout(h)

        return self.out(h).squeeze(-1)


    @torch.no_grad()
    def explain_nodes(
        self,
        x_all: torch.Tensor,
        node_indices: np.ndarray,
        feat_cols: list,
        dates: list,
        topk_feats: int = 5,
    ):
        """
        Memory-safe explanations: compute hist/att ONLY for node_indices.
        Returns list[dict] matching your original output schema.
        """
        self.eval()
        device = x_all.device
        idx = torch.tensor(node_indices, device=device, dtype=torch.long)

        time_idx_np = self.time_idx[idx].detach().cpu().numpy().astype(int)
        local_nodes_np = self.local_node_idx[idx].detach().cpu().numpy().astype(int)

        groups = [
            ("fast", self.fast_idx, self.fast_lag_offsets, self.w_fast),
            ("med", self.med_idx, self.med_lag_offsets, self.w_med),
            ("slow", self.slow_idx, self.slow_lag_offsets, self.w_slow),
        ]

        explanations = []
        for bi in range(idx.numel()):
            nid = int(idx[bi].item())
            t0 = int(time_idx_np[bi])
            node0 = int(local_nodes_np[bi])

            candidates = []

            for gname, gidx, lags, wparam in groups:
                if gidx.numel() == 0 or lags.numel() == 0:
                    continue

                hist = self._build_group_history_slice(
                    x_all=x_all,
                    group_idx=gidx.to(device),
                    lag_offsets=lags.to(device),
                    idx_slice=torch.tensor([nid], device=device, dtype=torch.long),
                )  # [1,F,L]
                if hist.numel() == 0:
                    continue

                hist = hist[0]  # [F,L]
                w = wparam.to(device=device, dtype=hist.dtype)  # [F,L]
                s = hist * w
                s_flat = s.view(1, -1)  # [1,F*L]
                att_flat = self._stable_softmax_1d(s_flat).view_as(s)  # [F,L]

                vals = hist.detach().cpu().numpy()
                atts = att_flat.detach().cpu().numpy()
                contrib = np.abs(vals * atts)

                Fg, Lg = contrib.shape
                gidx_np = gidx.detach().cpu().numpy()
                lags_np = lags.detach().cpu().numpy()

                for f in range(Fg):
                    for l in range(Lg):
                        score = float(contrib[f, l])
                        if score <= 0.0:
                            continue

                        lag_days = int(lags_np[l])
                        t_lag = t0 - lag_days
                        if t_lag < 0 or t_lag >= len(dates):
                            continue

                        f_global = int(gidx_np[f])
                        feat_name = feat_cols[f_global]

                        candidates.append(
                            dict(
                                group=gname,
                                feature=str(feat_name),
                                value=float(vals[f, l]),
                                lag_days=int(lag_days),
                                lag_date=str(dates[t_lag]),
                                attention=float(atts[f, l]),
                                contribution=score,
                            )
                        )

            candidates_sorted = sorted(candidates, key=lambda d: -d["contribution"])
            top_features = candidates_sorted[:topk_feats]

            explanations.append(
                dict(
                    nid=nid,
                    local_node=node0,
                    time_index=t0,
                    date=str(dates[t0]),
                    top_features=top_features,
                )
            )

        return explanations


# ---------------------------------------------------------------------------
# Metrics + threshold selection
# ---------------------------------------------------------------------------

def compute_basic_metrics(y_true, prob, thr=0.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    mask = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[mask]
    prob = prob[mask]

    if len(y_true) == 0:
        return {"n": 0, "pos_rate": np.nan, "roc_auc": np.nan, "pr_auc": np.nan,
                "f1": np.nan, "acc": np.nan, "prec": np.nan, "rec": np.nan, "thr_f1": thr}

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
    prec = (y_true[y_pred == 1].sum() / y_pred.sum()) if y_pred.sum() > 0 else 0.0
    rec = (y_pred[y_true == 1].sum() / y_true.sum()) if y_true.sum() > 0 else 0.0

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


def compute_early_warning_stats(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_idx: np.ndarray,
    thr: float,
    window_days: int = 5,
) -> EarlyWarningStats:
    y_true = y_true.astype(int)
    y_score = np.asarray(y_score, dtype=float)
    time_idx = np.asarray(time_idx)

    event_idx = np.where(y_true == 1)[0]
    n_events = int(event_idx.size)

    alerts = (y_score >= thr)
    lead_times = []
    hits = 0

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
            events=0, hit_rate=float("nan"), miss_rate=float("nan"),
            mean_lead=float("nan"), median_lead=float("nan"),
            window_days=window_days, thr=thr,
            fa_rate=fa_rate, fa_count=fa_count, fa_per_event=float("nan"),
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


@dataclass
class ThresholdSelectionResult:
    best_thr_f1: float
    best_val_f1: float
    best_thr_ew: float
    best_val_ew_hit_rate: float
    best_val_ew_fa_rate: float
    best_val_ew_fa_per_event: float
    c_miss: float
    c_false: float
    min_hit_for_ew: float


def sweep_thresholds(
    y_true_val: np.ndarray,
    y_score_val: np.ndarray,
    time_val: np.ndarray,
    window_days: int = 5,
    c_miss: float = 1.0,
    c_false: float = 0.05,
    n_thresholds: int = 1001,
    min_hit_for_ew: float = 0.90,
) -> ThresholdSelectionResult:
    y_true_val = y_true_val.astype(int)
    y_score_val = np.asarray(y_score_val, dtype=float)
    time_val = np.asarray(time_val)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    best_f1 = -1.0
    best_thr_f1 = 0.5

    best_thr_ew = 0.5
    best_hit_rate_ew = 0.0
    best_fa_rate_ew = 1.0
    best_fa_per_event_ew = float("inf")

    fallback_thr = 0.5
    fallback_hit = -1.0
    fallback_fa_per_event = float("inf")

    for thr in thresholds:
        node_metrics = compute_basic_metrics(y_true_val, y_score_val, thr)
        f1 = node_metrics["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr_f1 = thr

        ew_stats = compute_early_warning_stats(y_true_val, y_score_val, time_val, thr, window_days)
        hit_rate = ew_stats.hit_rate
        fa_rate = ew_stats.fa_rate
        fa_per_event = ew_stats.fa_per_event

        if hit_rate > fallback_hit or (math.isclose(hit_rate, fallback_hit) and fa_per_event < fallback_fa_per_event):
            fallback_hit = hit_rate
            fallback_fa_per_event = fa_per_event
            fallback_thr = thr

        if hit_rate >= min_hit_for_ew:
            if fa_per_event < best_fa_per_event_ew:
                best_fa_per_event_ew = fa_per_event
                best_thr_ew = thr
                best_hit_rate_ew = hit_rate
                best_fa_rate_ew = fa_rate

    if best_fa_per_event_ew == float("inf"):
        best_thr_ew = fallback_thr
        best_hit_rate_ew = fallback_hit
        ew_stats = compute_early_warning_stats(y_true_val, y_score_val, time_val, best_thr_ew, window_days)
        best_fa_rate_ew = ew_stats.fa_rate
        best_fa_per_event_ew = ew_stats.fa_per_event

    return ThresholdSelectionResult(
        best_thr_f1=float(best_thr_f1),
        best_val_f1=float(best_f1),
        best_thr_ew=float(best_thr_ew),
        best_val_ew_hit_rate=float(best_hit_rate_ew),
        best_val_ew_fa_rate=float(best_fa_rate_ew),
        best_val_ew_fa_per_event=float(best_fa_per_event_ew),
        c_miss=float(c_miss),
        c_false=float(c_false),
        min_hit_for_ew=float(min_hit_for_ew),
    )


def compute_highwind_rank_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    wind: np.ndarray,
    time_idx: np.ndarray,
    wind_percentile: float = 95.0,
    K_list=(5, 10, 20),
):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    wind = np.asarray(wind, dtype=float)
    time_idx = np.asarray(time_idx, dtype=int)

    if y_true.size == 0:
        return {"precision_at_k": {}, "recall_at_k": {}}

    thr_w = np.percentile(wind, wind_percentile)
    mask_hw = (wind >= thr_w)
    if not np.any(mask_hw):
        return {"precision_at_k": {}, "recall_at_k": {}}

    y_hw = y_true[mask_hw]
    p_hw = prob[mask_hw]
    t_hw = time_idx[mask_hw]

    Ks = sorted(set(K_list))
    tp_K = {K: 0.0 for K in Ks}
    fp_K = {K: 0.0 for K in Ks}
    pos_K = {K: 0.0 for K in Ks}

    for t in np.unique(t_hw):
        idx_t = np.where(t_hw == t)[0]
        if idx_t.size == 0:
            continue
        scores_t = p_hw[idx_t]
        labels_t = y_hw[idx_t]
        order = np.argsort(-scores_t)

        for K in Ks:
            m = min(K, idx_t.size)
            sel_local = order[:m]
            labels_sel = labels_t[sel_local]  # FIXED (was wrong indexing)
            tp = labels_sel.sum()
            fp = m - tp
            tp_K[K] += float(tp)
            fp_K[K] += float(fp)
            pos_K[K] += float(labels_t.sum())

    prec_at_k = {}
    rec_at_k = {}
    for K in Ks:
        denom_p = tp_K[K] + fp_K[K]
        prec_at_k[K] = float(tp_K[K] / denom_p) if denom_p > 0 else 0.0
        rec_at_k[K] = float(tp_K[K] / pos_K[K]) if pos_K[K] > 0 else 0.0

    return {"precision_at_k": prec_at_k, "recall_at_k": rec_at_k, "wind_percentile": float(wind_percentile)}


def find_low_fa_threshold(
    y_true_val: np.ndarray,
    y_score_val: np.ndarray,
    time_val: np.ndarray,
    window_days: int = 5,
    target_hit: float = 0.80,
    c_miss: float = 1.0,
    c_false: float = 1.0,
    target_fa_rate: float = -1.0,
    n_thresholds: int = 1001,
) -> LowFAThresholdResult:
    y_true_val = y_true_val.astype(int)
    y_score_val = np.asarray(y_score_val, dtype=float)
    time_val = np.asarray(time_val)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    rows = []
    for thr in thresholds:
        ew = compute_early_warning_stats(y_true_val, y_score_val, time_val, thr, window_days)
        hit = float(ew.hit_rate)
        miss = float(ew.miss_rate)
        fa_rate = float(ew.fa_rate)
        fa_per_event = float(ew.fa_per_event)
        cost = float(c_miss * miss + c_false * fa_rate)
        rows.append((float(thr), hit, miss, fa_rate, fa_per_event, cost))

    finite_rows = [r for r in rows if np.isfinite(r[3]) and np.isfinite(r[1])]
    if not finite_rows:
        return LowFAThresholdResult(
            thr=0.5, hit_rate=0.0, miss_rate=1.0,
            fa_rate=float("nan"), fa_per_event=float("nan"), cost=float("inf")
        )

    # 0) Legacy behavior (pre-change): maximize hit fallback, then minimize FA/event
    # This path is used when no explicit FA cap is requested.
    if target_fa_rate <= 0:
        best_thr = 0.5
        best_hit = -1.0
        best_miss = 1.0
        best_fa_rate = 1.0
        best_fa_per_event = float("inf")

        fallback_thr = 0.5
        fallback_hit = -1.0
        fallback_miss = 1.0
        fallback_fa_per_event = float("inf")
        fallback_fa_rate = 1.0

        for thr, hit, miss, fa_rate, fa_per_event, _cost in finite_rows:
            if hit > fallback_hit or (math.isclose(hit, fallback_hit) and fa_per_event < fallback_fa_per_event):
                fallback_hit = hit
                fallback_miss = miss
                fallback_fa_per_event = fa_per_event
                fallback_fa_rate = fa_rate
                fallback_thr = thr

            if hit >= target_hit:
                if fa_per_event < best_fa_per_event:
                    best_thr = thr
                    best_hit = hit
                    best_miss = miss
                    best_fa_rate = fa_rate
                    best_fa_per_event = fa_per_event

        if best_fa_per_event == float("inf"):
            best_thr = fallback_thr
            best_hit = fallback_hit
            best_miss = fallback_miss
            best_fa_rate = fallback_fa_rate
            best_fa_per_event = fallback_fa_per_event

        best_cost = float(c_miss * best_miss + c_false * best_fa_rate)
        return LowFAThresholdResult(
            thr=float(best_thr),
            hit_rate=float(best_hit),
            miss_rate=float(best_miss),
            fa_rate=float(best_fa_rate),
            fa_per_event=float(best_fa_per_event),
            cost=float(best_cost),
        )

    # 1) Optional hard FA cap + hit constraint
    if target_fa_rate > 0:
        feasible_cap = [r for r in finite_rows if (r[1] >= target_hit and r[3] <= target_fa_rate)]
        if feasible_cap:
            # FA-first, then maximize hit, then prefer higher threshold (fewer alerts)
            best = min(feasible_cap, key=lambda r: (r[3], -r[1], -r[0], r[4]))
            return LowFAThresholdResult(
                thr=best[0], hit_rate=best[1], miss_rate=best[2],
                fa_rate=best[3], fa_per_event=best[4], cost=best[5]
            )
        # If hit-floor is infeasible, still honor FA cap if possible
        feasible_fa_only = [r for r in finite_rows if r[3] <= target_fa_rate]
        if feasible_fa_only:
            # maximize hit under cap, then minimize FA, then prefer higher threshold
            best = min(feasible_fa_only, key=lambda r: (-r[1], r[3], -r[0], r[4]))
            return LowFAThresholdResult(
                thr=best[0], hit_rate=best[1], miss_rate=best[2],
                fa_rate=best[3], fa_per_event=best[4], cost=best[5]
            )

    # 2) FA-first under hit floor
    feasible_hit = [r for r in finite_rows if r[1] >= target_hit]
    if feasible_hit:
        best = min(feasible_hit, key=lambda r: (r[3], -r[1], -r[0], r[4]))
        return LowFAThresholdResult(
            thr=best[0], hit_rate=best[1], miss_rate=best[2],
            fa_rate=best[3], fa_per_event=best[4], cost=best[5]
        )

    # 3) Fallback: minimize weighted cost if hit floor cannot be met
    best = min(finite_rows, key=lambda r: (r[5], r[3], -r[1], -r[0]))
    return LowFAThresholdResult(
        thr=best[0], hit_rate=best[1], miss_rate=best[2],
        fa_rate=best[3], fa_per_event=best[4], cost=best[5]
    )


@torch.no_grad()
def evaluate_classifier(
    model,
    data,
    device,
    model_kind="sage",
    window_days: int = 5,
    c_miss: float = 1.0,
    c_false: float = 0.05,
    lowfa_target_hit: float = 0.80,
    target_fa_rate: float = -1.0,
    n_thresholds: int = 1001,
):
    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    ei_spat = data.edge_index_spat.to(device)
    ei_temp = data.edge_index_temp.to(device)
    wind = data.wind.cpu().numpy()
    time_all = data.time_idx.cpu().numpy()

    if model_kind == "gdn":
        logits, _ = model(x, ei_spat, ei_temp)
    else:
        logits = model(x, ei_spat, ei_temp)

    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    prob = torch.sigmoid(logits).cpu().numpy()
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_np = y.cpu().numpy()
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0)

    masks = {
        "train": data.train_mask.cpu().numpy(),
        "val": data.val_mask.cpu().numpy(),
        "test": data.test_mask.cpu().numpy(),
    }

    results = {"splits": {}, "highwind": {}}
    results["raw"] = {
        "prob": prob,
        "y": y_np,
        "time_idx": data.time_idx.cpu().numpy(),
        "local_node_idx": data.local_node_idx.cpu().numpy(),
        "train_mask": masks["train"],
        "val_mask": masks["val"],
        "test_mask": masks["test"],
        "x": data.x.cpu().numpy(),
        "gtg_Tmax": getattr(data, "gtg_Tmax", -1),
        "feat_cols": np.array(data.feat_cols),
        "dates": np.array(data.dates),
    }

    val_mask = masks["val"]
    thr_res = sweep_thresholds(
        y_true_val=y_np[val_mask],
        y_score_val=prob[val_mask],
        time_val=time_all[val_mask],
        window_days=window_days,
        c_miss=c_miss,
        c_false=c_false,
        n_thresholds=n_thresholds,
        min_hit_for_ew=0.90,
    )

    thr_f1 = thr_res.best_thr_f1
    thr_ew = thr_res.best_thr_ew

    if model_kind == "gtg_tgcn":
        lowfa_res = find_low_fa_threshold(
            y_true_val=y_np[masks["val"]],
            y_score_val=prob[masks["val"]],
            time_val=time_all[masks["val"]],
            window_days=window_days,
            target_hit=lowfa_target_hit,
            c_miss=c_miss,
            c_false=c_false,
            target_fa_rate=target_fa_rate,
            n_thresholds=n_thresholds,
        )
        print(
            f"[GTG-TGCN low-FA threshold] thr={lowfa_res.thr:.4f}, "
            f"hit_rate_val={lowfa_res.hit_rate:.3f}, "
            f"FA_rate_val={lowfa_res.fa_rate:.5f}, "
            f"FA_per_event_val={lowfa_res.fa_per_event:.1f}, "
            f"cost_val={lowfa_res.cost:.5f} "
            f"(c_miss={c_miss:.3f}, c_false={c_false:.3f}, "
            f"target_hit={lowfa_target_hit:.2f}, target_fa={target_fa_rate:.3f})"
        )
        thr_ew = lowfa_res.thr

    results["best_thr_f1"] = float(thr_f1)
    results["best_thr_ew"] = float(thr_ew)
    results["val_f1_at_best_thr"] = float(thr_res.best_val_f1)
    results["val_ew_hit_rate"] = float(thr_res.best_val_ew_hit_rate)
    results["val_ew_fa_rate"] = float(thr_res.best_val_ew_fa_rate)

    print(
        f"[val thresholds] best_thr_f1={thr_res.best_thr_f1:.4f} (F1={thr_res.best_val_f1:.4f}), "
        f"best_thr_ew={thr_ew:.4f} (min_hit_for_ew={thr_res.min_hit_for_ew:.2f})"
    )

    for split_name, m in masks.items():
        res_split = compute_basic_metrics(y_np[m], prob[m], thr=thr_f1)
        results["splits"][split_name] = res_split
        print(
            f"[{split_name}] n={res_split['n']} pos_rate={res_split['pos_rate']:.6f} "
            f"ROC-AUC={res_split['roc_auc']:.4f} PR-AUC={res_split['pr_auc']:.4f} "
            f"F1@thr_f1={res_split['f1']:.4f} thr_f1={thr_f1:.4f}"
        )

    # High-wind metrics on TEST
    m_test = masks["test"]
    y_test = y_np[m_test]
    p_test = prob[m_test]
    w_test = wind[m_test]
    t_test = time_all[m_test]

    if len(y_test) > 0 and np.any(w_test != 0.0):
        for p in [90, 95, 98, 99]:
            thr_w = np.percentile(w_test, p)
            hw_mask = (w_test >= thr_w)
            y_hw = y_test[hw_mask]
            p_hw = p_test[hw_mask]
            res_hw = compute_basic_metrics(y_hw, p_hw, thr=thr_f1)
            key = f"p{p}"
            results["highwind"][key] = res_hw
            print(
                f"[test highwind {key}] n={res_hw['n']} pos_rate={res_hw['pos_rate']:.6f} "
                f"ROC-AUC={res_hw['roc_auc']:.4f} PR-AUC={res_hw['pr_auc']:.4f} "
                f"F1@thr_f1={res_hw['f1']:.4f}"
            )

            rank_hw = compute_highwind_rank_metrics(
                y_true=y_test,
                prob=p_test,
                wind=w_test,
                time_idx=t_test,
                wind_percentile=p,
                K_list=(5, 10, 20),
            )
            results["highwind"][f"{key}_rank"] = rank_hw

    ew_stats = compute_early_warning_stats(y_test, p_test, t_test, thr_ew, window_days)
    results["early_warning_5d"] = {
        "events": ew_stats.events,
        "hit_rate": ew_stats.hit_rate,
        "miss_rate": ew_stats.miss_rate,
        "mean_lead": ew_stats.mean_lead,
        "median_lead": ew_stats.median_lead,
        "window_days": ew_stats.window_days,
        "thr": ew_stats.thr,
        "fa_rate": ew_stats.fa_rate,
        "fa_count": ew_stats.fa_count,
        "fa_per_event": ew_stats.fa_per_event,
    }
    print(
        f"[test early_warning_{window_days}d@thr_ew] events={ew_stats.events} "
        f"hit_rate={ew_stats.hit_rate:.3f} miss_rate={ew_stats.miss_rate:.3f} "
        f"mean_lead={ew_stats.mean_lead:.2f}d median_lead={ew_stats.median_lead:.2f}d "
        f"FA_rate={ew_stats.fa_rate:.5f} FA_per_event={ew_stats.fa_per_event:.2f} thr_ew={thr_ew:.4f}"
    )

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    data,
    device,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-5,
    model_kind="sage",
    window_days: int = 5,
    c_miss: float = 1.0,
    c_false: float = 0.05,
    lowfa_target_hit: float = 0.80,
    target_fa_rate: float = -1.0,
    n_thresholds: int = 1001,
    pos_weight_power: float = 1.0,
    pos_weight_max: float = 0.0,
    use_amp: bool = True,
):
    model.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    ei_spat = data.edge_index_spat.to(device)
    ei_temp = data.edge_index_temp.to(device)
    train_mask = data.train_mask.to(device)

    with torch.no_grad():
        y_train = torch.nan_to_num(y[train_mask], nan=0.0, posinf=1.0, neginf=0.0)
        pos = y_train.sum().item()
        neg = y_train.numel() - pos
        raw_pos_weight = (neg / (pos + 1e-8)) if pos > 0 else 1.0
        scaled_pos_weight = raw_pos_weight ** float(max(pos_weight_power, 0.0))
        if pos_weight_max > 0:
            scaled_pos_weight = min(scaled_pos_weight, float(pos_weight_max))
        pos_weight = torch.tensor([scaled_pos_weight], device=device)

    print(
        f"Estimating class imbalance... pos_rate={pos / (pos + neg + 1e-8):.6f}, "
        f"raw_pos_weight={raw_pos_weight:.3f}, "
        f"scaled_pos_weight={pos_weight.item():.3f} "
        f"(power={pos_weight_power:.2f}, max={pos_weight_max:.1f})"
    )

    if model_kind == "gdn":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    amp_enabled = bool(use_amp and device.type == "cuda")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if model_kind == "gdn":
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, recon = model(x, ei_spat, ei_temp)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                recon = torch.nan_to_num(recon, nan=0.0, posinf=10.0, neginf=-10.0)
                loss = model.loss(logits[train_mask], recon[train_mask], x[train_mask], y[train_mask], pos_weight=pos_weight)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x, ei_spat, ei_temp)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                y_clean = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                loss = criterion(logits[train_mask], y_clean[train_mask])

        if not torch.isfinite(loss):
            print(f"Epoch {epoch:03d} | NON-FINITE LOSS ({loss.item()}); skipping step.")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch {epoch:03d} | train_loss={loss.item():.6f}")

    results = evaluate_classifier(
        model,
        data,
        device,
        model_kind=model_kind,
        window_days=window_days,
        c_miss=c_miss,
        c_false=c_false,
        lowfa_target_hit=lowfa_target_hit,
        target_fa_rate=target_fa_rate,
        n_thresholds=n_thresholds,
    )
    return results


# ---------------------------------------------------------------------------
# Explanation (fixed to avoid full-N attention tensors)
# ---------------------------------------------------------------------------

@torch.no_grad()
def explain_gtg_tgcn_top_nodes(
    model: GTG_TGCN_Classifier,
    data: Data,
    device,
    split: str = "test",
    thr: float = 0.5,
    max_examples: int = 20,
    topk_feats: int = 5,
    max_per_date: int = 3,
    label_filter: str = "pos",  # pos | neg | all
):
    model.eval()
    model.to(device)

    x = data.x.to(device)
    ei_spat = data.edge_index_spat.to(device)
    ei_temp = data.edge_index_temp.to(device)

    logits = model(x, ei_spat, ei_temp)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    y = data.y.cpu().numpy().astype(float)
    time_idx_all = data.time_idx.cpu().numpy()
    dates = data.dates

    if split == "train":
        mask = data.train_mask.cpu().numpy()
    elif split == "val":
        mask = data.val_mask.cpu().numpy()
    else:
        mask = data.test_mask.cpu().numpy()

    idx_split = np.where(mask)[0]
    idx_cand = idx_split
    if np.isfinite(thr):
        idx_cand = idx_cand[probs[idx_cand] >= thr]

    if label_filter == "pos":
        idx_cand = idx_cand[y[idx_cand] == 1]
    elif label_filter == "neg":
        idx_cand = idx_cand[y[idx_cand] == 0]
    elif label_filter == "all":
        pass
    else:
        raise ValueError(f"Unknown label_filter='{label_filter}'. Use one of: pos, neg, all.")

    if idx_cand.size == 0:
        print(f"[explain_gtg_tgcn_top_nodes] No nodes in split='{split}' above thr={thr:.3f}")
        return []

    idx_sorted = idx_cand[np.argsort(-probs[idx_cand])]
    print(
        f"[explain_gtg_tgcn_top_nodes] split={split} label_filter={label_filter} "
        f"candidates={idx_sorted.size}"
    )

    selected = []
    if max_per_date > 0:
        # diversify by date
        per_date_count = defaultdict(int)
        for nid in idx_sorted:
            t0 = int(time_idx_all[nid])
            d = dates[t0]
            if per_date_count[d] >= max_per_date:
                continue
            selected.append(int(nid))
            per_date_count[d] += 1
            if max_examples > 0 and len(selected) >= max_examples:
                break
    else:
        selected = [int(nid) for nid in idx_sorted]
        if max_examples > 0:
            selected = selected[:max_examples]

    if not selected:
        return []

    expl = model.explain_nodes(
        x_all=x,
        node_indices=np.array(selected, dtype=np.int64),
        feat_cols=data.feat_cols,
        dates=data.dates,
        topk_feats=topk_feats,
    )

    # add label/prob fields (kept identical to your prior output)
    for e in expl:
        nid = e["nid"]
        e["label"] = float(y[nid])
        e["prob"] = float(probs[nid])

    return expl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to daily CSV")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind-feature", type=str, default="wind_speed")
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--c-miss", type=float, default=1.0)
    parser.add_argument("--c-false", type=float, default=0.05)
    parser.add_argument("--lowfa-target-hit", type=float, default=0.80)
    parser.add_argument("--target-fa-rate", type=float, default=-1.0, help=">0 enforces FA-rate cap on VAL")
    parser.add_argument("--n-thresholds", type=int, default=1001)
    parser.add_argument("--pos-weight-power", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=0.0, help="<=0 disables max cap")
    parser.add_argument("--out-prefix", type=str, default="gtg_tgcn")
    parser.add_argument("--skip-explanations", action="store_true", help="Skip post-training explanation generation.")
    parser.add_argument("--explain-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--explain-thr", type=float, default=float("nan"), help="If set, use this probability threshold for selecting explanation candidates. Use -1 to include all.")
    parser.add_argument("--explain-max-examples", type=int, default=20, help="Max number of explained nodes. <=0 means all candidates.")
    parser.add_argument("--explain-topk-feats", type=int, default=5, help="Top feature-lag contributors per node explanation.")
    parser.add_argument("--explain-max-per-date", type=int, default=3, help="Per-date cap for diversity. <=0 disables date cap.")
    parser.add_argument("--explain-label-filter", type=str, default="pos", choices=["pos", "neg", "all"], help="Which labels to explain among selected candidates.")

    # NEW: GTG horizon + chunk config
    parser.add_argument("--gtg-years", type=float, default=12.0)
    parser.add_argument("--total-years", type=float, default=20.0)
    parser.add_argument("--gtg-chunk", type=int, default=50_000)
    parser.add_argument("--no-gtg-restrict", action="store_true", help="Disable GTG horizon restriction (not recommended).")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision.")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = load_daily_csv(args.csv)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Out prefix: {args.out_prefix}")

    print("Building graph WITHOUT wind edges (self temporal edges only)...")
    data_nowind = build_space_time_graph(
        df,
        use_wind_edges=False,
        wind_feature_name=args.wind_feature,
        restrict_to_gtg_horizon=(not args.no_gtg_restrict),
        gtg_years=args.gtg_years,
        total_years=args.total_years,
    )

    in_dim = data_nowind.x.size(1)

    # Feature groups for GTG
    fast_feats = [
        "v10", "wind_speed", "merra_dust_anom", "wind_speed_anom", "wind_speed_mean_7d",
        "wind_speed_nb_mean", "u10", "wind_dir_cos", "wind_dir_sin", "erosion_index_2",
        "NDVI_trend_30d", "NDVI_mean_30d", "merra_dust", "Albedo_anom", "Albedo_nb_mean",
        "Albedo_contrast", "wind_speed_contrast", "Evapo_sum_30d", "water_deficit_30d",
        "Precip_sum_30d", "Albedo", "wind_speed_max_3d", "erosion_index_1", "days_since_rain",
    ]
    med_feats = ["SAVI", "NDVI_anom", "AAT_anom", "SPEI3_anom", "NDVI_nb_mean", "NDVI_contrast", "NDVI"]
    slow_feats = ["SPEI3", "AAT", "Evapotransp", "Precip"]

    feat_cols = data_nowind.feat_cols
    name_to_idx = {name: i for i, name in enumerate(feat_cols)}

    fast_idx = [name_to_idx[n] for n in fast_feats if n in name_to_idx]
    med_idx = [name_to_idx[n] for n in med_feats if n in name_to_idx]
    slow_idx = [name_to_idx[n] for n in slow_feats if n in name_to_idx]

    print(f"#fast={len(fast_idx)}, #medium={len(med_idx)}, #slow={len(slow_idx)}")

    experiments = [
        (
            "gtg_tgcn_nowind",
            "gtg_tgcn",
            GTG_TGCN_Classifier(
                in_dim=in_dim,
                fast_idx=fast_idx,
                med_idx=med_idx,
                slow_idx=slow_idx,
                node_t_to_nid=data_nowind.node_t_to_nid,
                local_node_idx=data_nowind.local_node_idx,
                time_idx=data_nowind.time_idx,
                gtg_chunk=args.gtg_chunk,
            ),
            data_nowind,
        )
    ]

    all_results = {}

    for key, model_kind, model, data in experiments:
        print("=" * 80)
        print(f"Experiment: {key}")
        print("=" * 80)

        res = train_model(
            model,
            data,
            device=device,
            epochs=args.epochs,
            lr=1e-3,
            weight_decay=1e-5,
            model_kind=model_kind,
            window_days=args.window_days,
            c_miss=args.c_miss,
            c_false=args.c_false,
            lowfa_target_hit=args.lowfa_target_hit,
            target_fa_rate=args.target_fa_rate,
            n_thresholds=args.n_thresholds,
            pos_weight_power=args.pos_weight_power,
            pos_weight_max=args.pos_weight_max,
            use_amp=(not args.no_amp),
        )
        all_results[key] = res

        if model_kind == "gtg_tgcn":
            raw = res.get("raw")
            if raw is not None:
                npz_path = f"{args.out_prefix}_rl_data.npz"
                np.savez(npz_path, **raw)
                print(f"Saved GTG-TGCN outputs for RL to {npz_path}")

        print(f"[{key}] summary:")
        print(f"  best_thr_f1={res['best_thr_f1']:.4f}, val_f1_at_best_thr={res['val_f1_at_best_thr']:.4f}")
        print(f"  best_thr_ew={res['best_thr_ew']:.4f}, val_ew_hit_rate={res['val_ew_hit_rate']:.3f}, val_ew_FA_rate={res['val_ew_fa_rate']:.5f}")

        # Explanations (memory safe)
        if model_kind == "gtg_tgcn" and (not args.skip_explanations):
            # Use operational alert threshold for explanations (usually less brittle than F1 threshold).
            if np.isfinite(args.explain_thr):
                thr_for_expl = float(args.explain_thr)
                thr_src = "cli"
            else:
                thr_for_expl = float(res.get("best_thr_ew", res["best_thr_f1"]))
                thr_src = "best_thr_ew"

            thr_desc = f"{thr_for_expl:.3f}" if np.isfinite(thr_for_expl) else "no-threshold"
            print(
                f"\n[GTG-TGCN] Generating explanations "
                f"(split={args.explain_split}, label={args.explain_label_filter}, "
                f"thr={thr_desc} from {thr_src}, max_examples={args.explain_max_examples}, "
                f"topk_feats={args.explain_topk_feats}, max_per_date={args.explain_max_per_date})"
            )

            expl = explain_gtg_tgcn_top_nodes(
                model=model,
                data=data,
                device=device,
                split=args.explain_split,
                thr=thr_for_expl,
                max_examples=args.explain_max_examples,
                topk_feats=args.explain_topk_feats,
                max_per_date=args.explain_max_per_date,
                label_filter=args.explain_label_filter,
            )

            import json
            expl_path = f"{args.out_prefix}_explanations.json"
            with open(expl_path, "w", encoding="utf-8") as f:
                json.dump(expl, f, ensure_ascii=False, indent=2)

            print(f"[GTG-TGCN] Saved {len(expl)} explanations to {expl_path}\n")

    print("\n=== Global summary (test split) ===")
    for key, res in all_results.items():
        test_stats = res["splits"]["test"]
        ew = res["early_warning_5d"]
        print(
            f"{key:20s} | ROC-AUC={test_stats['roc_auc']:.4f} "
            f"PR-AUC={test_stats['pr_auc']:.4f} F1={test_stats['f1']:.4f} "
            f"| hit_rate_{args.window_days}d={ew['hit_rate']:.3f} "
            f"mean_lead_{args.window_days}d={ew['mean_lead']:.2f}d "
            f"FA_rate={ew['fa_rate']:.5f}"
        )


if __name__ == "__main__":
    main()
