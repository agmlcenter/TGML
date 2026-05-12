#!/usr/bin/env python3
"""
Space–Time Graph ablation with optional wind-directed temporal edges.

Models (node-level classifiers):
  1) GraphWaveNet-style residual GraphConv network  -> wavenet_*
  2) Temporal GAT over spatio-temporal graph        -> tgat_*
  3) Graph Transformer (TransformerConv)            -> gtr_*

We run each model in two modes:
  - *_nowind : temporal edges = self-edges only (t-1 -> same node at t)
  - *_wind   : temporal edges use wind_dir_sin / wind_dir_cos to route
               downwind edges plus self-edge.

Loss: BCE with pos_weight (no focal, no reconstruction).

Inputs:
  Daily CSV with at least:
    node, date, label, wind_speed, wind_dir_sin, wind_dir_cos, ...
  'node' is a string "row_col" on a rectangular grid (e.g. "7_4").
"""

import argparse
import math
from collections import defaultdict
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, GATConv, TransformerConv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Data loading and graph construction
# ---------------------------------------------------------------------------

def load_daily_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"node", "date", "label"}.issubset(df.columns), "CSV must have node,date,label"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "node"]).reset_index(drop=True)
    return df


def encode_grid_nodes(df: pd.DataFrame):
    """
    Parse node strings like '7_4' into (row, col) and infer grid size.
    Returns:
      node_to_idx: dict "row_col" -> local integer id (0..N_cells-1)
      idx_to_rc: list of (row, col) in 0-based grid coordinates
      grid_h, grid_w
    """
    unique_nodes = sorted(set(df["node"].unique()))
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
    """
    Standardize features to mean 0, std 1 (per column), robust to NaNs/Infs.
    NaNs/Infs are converted to finite values at the end.
    """
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std = std + 1e-8  # avoid division by zero

    Xn = (X - mean) / std
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=10.0, neginf=-10.0)
    return Xn, mean, std


def angle_to_direction(sin_val: float, cos_val: float):
    """
    Map wind_dir_sin, wind_dir_cos to one of 8 directions.
    Returns integer in [0..7] and (dr,dc).

    Direction indices:
      0: East, 1: NE, 2: North, 3: NW, 4: West, 5: SW, 6: South, 7: SE
    """
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
) -> Data:
    """
    Build a space–time graph:

    - Nodes: (node, date)
    - Features: all numeric columns except {node,date,label}
    - Label: df["label"]
    - Spatial edges: undirected 4-neighbour grid per time slice
    - Temporal edges:
        * If use_wind_edges:
            from (node, t-1) -> downwind neighbour at t (+ self-edge)
        * Else:
            self-edge only: (node, t-1) -> (same node, t)

    Splits:
      train/val/test by time index (70% / 15% / 15%).

    Also attaches:
      data.wind : per-node scalar for wind_feature_name (float32)
    """
    node_to_idx, idx_to_rc, grid_h, grid_w = encode_grid_nodes(df)

    exclude_cols = {"node", "date", "label"}
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    feat_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    print(f"Using {len(feat_cols)} feature columns.")
    if wind_feature_name not in df.columns:
        print(
            f"WARNING: wind_feature_name='{wind_feature_name}' not in CSV; "
            f"high-wind metrics will be disabled."
        )
    else:
        print(f"Using '{wind_feature_name}' as wind feature for high-wind percentiles.")

    # Map dates to time indices
    unique_dates = np.sort(df["date"].unique())
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
            # If duplicates per (node,time) exist, we keep the first.
            continue
        nid = len(nt_to_nid)
        nt_to_nid[key] = nid
        nid_to_nt.append(key)
        X_rows.append(row[feat_cols].values.astype(np.float32))
        Y_rows.append(float(row["label"]))
        if wind_feature_name in df.columns:
            wind_rows.append(float(row[wind_feature_name]))

    X = np.stack(X_rows, axis=0)
    y = np.array(Y_rows, dtype=np.float32)

    X_norm, mean, std = normalize_features(X)
    print(f"Built {len(nt_to_nid)} nodes across {T} time steps.")

    # Clean labels: NaN -> 0
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

    if wind_rows:
        wind_vals = np.array(wind_rows, dtype=np.float32)
        wind_vals = np.nan_to_num(wind_vals, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        wind_vals = np.zeros(len(nid_to_nt), dtype=np.float32)

    # Build spatial edges (4-neighbour grid) per time
    edge_spat_src = []
    edge_spat_dst = []

    t_to_nids = defaultdict(list)
    for nid, (local_idx, t) in enumerate(nid_to_nt):
        t_to_nids[t].append(nid)

    # Precompute local_idx -> list of neighbours in 4-neighbour grid
    neighbours_by_local = {}
    for li, (r, c) in enumerate(idx_to_rc):
        nb = []
        for (nr, nc) in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                for li2, (rr, cc) in enumerate(idx_to_rc):
                    if rr == nr and cc == nc:
                        nb.append(li2)
                        break
        neighbours_by_local[li] = nb

    for t in range(T):
        for nid in t_to_nids[t]:
            local_idx, _ = nid_to_nt[nid]
            for nb_local in neighbours_by_local[local_idx]:
                key2 = (nb_local, t)
                if key2 in nt_to_nid:
                    nid2 = nt_to_nid[key2]
                    edge_spat_src.append(nid)
                    edge_spat_dst.append(nid2)
                    edge_spat_src.append(nid2)
                    edge_spat_dst.append(nid)

    # Build temporal edges
    edge_temp_src = []
    edge_temp_dst = []

    try:
        sin_idx = feat_cols.index("wind_dir_sin")
        cos_idx = feat_cols.index("wind_dir_cos")
        has_wind_dir = True
    except ValueError:
        has_wind_dir = False
        print(
            "WARNING: wind_dir_sin / wind_dir_cos not found; "
            "temporal edges will be self-only regardless of use_wind_edges."
        )

    for t in range(1, T):
        for nid_t, (local_idx, t2) in enumerate(nid_to_nt):
            if t2 != t:
                continue
            r, c = idx_to_rc[local_idx]
            prev_key = (local_idx, t - 1)
            if prev_key not in nt_to_nid:
                continue
            prev_nid = nt_to_nid[prev_key]

            # Default: self temporal edge prev_nid -> nid_t
            dst_nid_main = nid_t

            if use_wind_edges and has_wind_dir:
                prev_feat = X_norm[prev_nid]
                sin_val = prev_feat[sin_idx]
                cos_val = prev_feat[cos_idx]
                _, (dr, dc) = angle_to_direction(sin_val, cos_val)
                nr, nc = r + dr, c + dc

                neighbour_local_idx = None
                for li2, (rr, cc) in enumerate(idx_to_rc):
                    if rr == nr and cc == nc:
                        neighbour_local_idx = li2
                        break

                if neighbour_local_idx is not None:
                    dst_key = (neighbour_local_idx, t)
                    if dst_key in nt_to_nid:
                        dst_nid_main = nt_to_nid[dst_key]

                # main downwind edge
                edge_temp_src.append(prev_nid)
                edge_temp_dst.append(dst_nid_main)
                # ensure self-edge too if different
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

    # Time-based train/val/test split
    N_dates = len(unique_dates)
    train_T = int(0.7 * N_dates)
    val_T = int(0.85 * N_dates)

    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(x.size(0), dtype=torch.bool)

    for nid, (local_idx, t) in enumerate(nid_to_nt):
        if t < train_T:
            train_mask[nid] = True
        elif t < val_T:
            val_mask[nid] = True
        else:
            test_mask[nid] = True

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
    return data


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GraphWaveNetBlock(nn.Module):
    """
    Deep residual GraphConv block for GraphWaveNet-style model.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.res = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels)
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        res = self.res(x)
        out = out + res
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out


class GraphWaveNetClassifier(nn.Module):
    """
    GraphWaveNet-style classifier:
      - several residual GraphConv blocks on combined edges
      - node-wise MLP head to logits.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(GraphWaveNetBlock(hidden_dim, hidden_dim, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for block in self.blocks:
            h = block(h, edge_index)
        logits = self.head(h).squeeze(-1)
        return logits


class TemporalGATLayer(nn.Module):
    """
    Temporal GAT layer over the spatio-temporal graph.
    Lightweight config for large graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,  # output dim = out_channels
            dropout=dropout,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat(x, edge_index)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class TemporalGATClassifier(nn.Module):
    """
    Multi-layer Temporal GAT over combined spatial+temporal edges.
    Reduced hidden size / heads to fit in GPU memory.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        layers = []
        for _ in range(num_layers):
            layers.append(
                TemporalGATLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        logits = self.head(h).squeeze(-1)
        return logits


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer using TransformerConv.
    Lightweight config.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.trans = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,  # output dim = out_channels
            dropout=dropout,
            beta=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.trans(x, edge_index, edge_attr=edge_attr)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class GraphTransformerClassifier(nn.Module):
    """
    Graph Transformer on combined spatial+temporal edges.
    Edge attributes are optional and off by default.
    Reduced hidden size / heads to fit memory.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
        use_edge_attr: bool = False,
    ):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        layers = []
        for _ in range(num_layers):
            layers.append(
                GraphTransformerLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        edge_attr = None  # no edge features for now
        h = torch.relu(self.lin_in(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr=edge_attr)
        logits = self.head(h).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Training / evaluation helpers
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
    return thr, f1[idx]


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


@torch.no_grad()
def evaluate_classifier(model, data, device):
    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    ei_spat = data.edge_index_spat.to(device)
    ei_temp = data.edge_index_temp.to(device)
    wind = data.wind.cpu().numpy()

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

    # Threshold from validation
    m_val = masks["val"]
    if m_val.sum() > 0:
        thr_best, f1_val = best_f1_threshold(y_np[m_val], prob[m_val])
    else:
        thr_best, f1_val = (0.5, float("nan"))

    results["best_thr"] = float(thr_best)
    results["val_f1_at_best_thr"] = float(f1_val)

    # Global metrics for each split
    for split_name, m in masks.items():
        res_split = compute_basic_metrics(y_np[m], prob[m], thr=thr_best)
        results["splits"][split_name] = res_split
        print(
            f"[{split_name}] n={res_split['n']} pos_rate={res_split['pos_rate']:.6f} "
            f"ROC-AUC={res_split['roc_auc']:.4f} PR-AUC={res_split['pr_auc']:.4f} "
            f"F1@thr={res_split['f1']:.4f} thr={thr_best:.4f}"
        )

    # High-wind percentiles on TEST split only
    m_test = masks["test"]
    y_test = y_np[m_test]
    p_test = prob[m_test]
    w_test = wind[m_test]

    if len(y_test) > 0 and np.any(w_test != 0.0):
        percentiles = [90, 95, 98, 99]
        for p in percentiles:
            thr_w = np.percentile(w_test, p)
            hw_mask = w_test >= thr_w
            y_hw = y_test[hw_mask]
            p_hw = p_test[hw_mask]
            res_hw = compute_basic_metrics(y_hw, p_hw, thr=thr_best)
            key = f"p{p}"
            results["highwind"][key] = res_hw
            print(
                f"[test highwind {key}] n={res_hw['n']} pos_rate={res_hw['pos_rate']:.6f} "
                f"ROC-AUC={res_hw['roc_auc']:.4f} PR-AUC={res_hw['pr_auc']:.4f} "
                f"F1@thr={res_hw['f1']:.4f} thr={thr_best:.4f}"
            )

    return results


def train_model(
    model,
    data,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
):
    model.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    ei_spat = data.edge_index_spat.to(device)
    ei_temp = data.edge_index_temp.to(device)
    train_mask = data.train_mask.to(device)

    # Class imbalance weighting
    with torch.no_grad():
        y_train = y[train_mask]
        y_train = torch.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
        pos = y_train.sum().item()
        neg = (y_train.numel() - pos)
        if pos > 0:
            pos_weight = torch.tensor([neg / (pos + 1e-8)], device=device)
        else:
            pos_weight = torch.tensor([1.0], device=device)
    print(
        f"Estimating class imbalance... pos_rate={pos / (pos + neg + 1e-8):.6f}, "
        f"pos_weight~{pos_weight.item():.3f}"
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, ei_spat, ei_temp)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        y_clean = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        loss = criterion(logits[train_mask], y_clean[train_mask])

        if not torch.isfinite(loss):
            print(f"Epoch {epoch:03d} | NON-FINITE LOSS ({loss.item()}); skipping step.")
        else:
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch:03d} | train_loss={loss.item():.6f}")

    results = evaluate_classifier(model, data, device)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to daily CSV")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--wind-feature", type=str, default="wind_speed")
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = load_daily_csv(args.csv)

    global_device = torch.device(args.device)
    print(f"Device: {global_device}")

    print("Building graph WITHOUT wind edges (self temporal edges only)...")
    data_nowind = build_space_time_graph(
        df,
        use_wind_edges=False,
        wind_feature_name=args.wind_feature,
    )

    print("Building graph WITH wind-directed temporal edges...")
    data_wind = build_space_time_graph(
        df,
        use_wind_edges=True,
        wind_feature_name=args.wind_feature,
    )

    in_dim = data_nowind.x.size(1)

    experiments = [
        ("wavenet_nowind", GraphWaveNetClassifier(in_dim=in_dim), data_nowind),
        ("wavenet_wind", GraphWaveNetClassifier(in_dim=in_dim), data_wind),
        ("tgat_nowind", TemporalGATClassifier(in_dim=in_dim), data_nowind),
        ("tgat_wind", TemporalGATClassifier(in_dim=in_dim), data_wind),
        ("gtr_nowind", GraphTransformerClassifier(in_dim=in_dim), data_nowind),
        ("gtr_wind", GraphTransformerClassifier(in_dim=in_dim), data_wind),
    ]

    all_results = {}

    for key, model, data in experiments:
        print("=" * 80)
        print(f"Experiment: {key}")
        print("=" * 80)

        # All use the same device; if still OOM on GAT/Transformer, we can
        # later force them to CPU only.
        res = train_model(
            model,
            data,
            device=global_device,
            epochs=args.epochs,
            lr=1e-3,
            weight_decay=1e-5,
        )
        all_results[key] = res

        print(f"[{key}] summary:")
        print(
            f"  best_thr={res['best_thr']:.4f}, "
            f"val_f1_at_best_thr={res['val_f1_at_best_thr']:.4f}"
        )
        for split, stats in res["splits"].items():
            print(
                f"  [{split}] ROC-AUC={stats['roc_auc']:.4f}, "
                f"PR-AUC={stats['pr_auc']:.4f}, F1={stats['f1']:.4f}, "
                f"ACC={stats['acc']:.4f}, PREC={stats['prec']:.4f}, "
                f"REC={stats['rec']:.4f}"
            )
        if res["highwind"]:
            for pkey, stats in res["highwind"].items():
                print(
                    f"  [test {pkey}] ROC-AUC={stats['roc_auc']:.4f}, "
                    f"PR-AUC={stats['pr_auc']:.4f}, F1={stats['f1']:.4f}, "
                    f"ACC={stats['acc']:.4f}, PREC={stats['prec']:.4f}, "
                    f"REC={stats['rec']:.4f}, n={stats['n']}"
                )

        # Explicit cleanup between experiments
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n=== Global summary (test split) ===")
    for key, res in all_results.items():
        test_stats = res["splits"]["test"]
        print(
            f"{key:15s} | ROC-AUC={test_stats['roc_auc']:.4f} "
            f"PR-AUC={test_stats['pr_auc']:.4f} F1={test_stats['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
