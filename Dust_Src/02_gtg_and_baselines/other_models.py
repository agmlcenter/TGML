#!/usr/bin/env python3
"""
Run baseline/non-GTG models on the exact same data pipeline and evaluation used by gtg.py.

Models included:
- simple_gcn
- simple_gat
- gdn
- simple_cnn
- simple_lstm
- sage (optional)
- graphwavenet (optional)

All models use:
- same CSV loading
- same graph construction/splits
- same threshold selection + early-warning metrics from gtg.py
"""

import argparse
import json
import math
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gtg import (
    load_daily_csv,
    build_space_time_graph,
    train_model,
    SAGEClassifier,
    SimpleGATClassifier,
    GDNStyleClassifier,
    GraphWaveNetClassifier,
)


class SimpleGCNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        edge_index = torch.cat([edge_index_spat, edge_index_temp], dim=1)
        h = torch.relu(self.lin_in(x))
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = self.dropout(h)
        return self.out(h).squeeze(-1)


class _HistoryModelBase(nn.Module):
    """
    Helper for sequence models that build per-node temporal windows
    from (local_node_idx, time_idx, node_t_to_nid) without changing data splits.
    """

    def __init__(
        self,
        node_t_to_nid: torch.Tensor,
        local_node_idx: torch.Tensor,
        time_idx: torch.Tensor,
        lookback: int = 14,
        hist_chunk: int = 50_000,
    ):
        super().__init__()
        self.register_buffer("node_t_to_nid", node_t_to_nid.long())
        self.register_buffer("local_node_idx", local_node_idx.long())
        self.register_buffer("time_idx", time_idx.long())
        self.lookback = int(lookback)
        self.hist_chunk = int(hist_chunk)

    @staticmethod
    def _hist_dtype(x_all: torch.Tensor):
        if x_all.is_cuda:
            return torch.float16
        return x_all.dtype

    def _build_history_slice(self, x_all: torch.Tensor, idx_slice: torch.Tensor) -> torch.Tensor:
        """
        Returns history tensor [B, L, F] where L=lookback and latest step is at L-1.
        Missing history is zero-padded.
        """
        device = x_all.device
        idx_slice = idx_slice.to(device)
        local_nodes = self.local_node_idx[idx_slice].to(device)
        time_idx = self.time_idx[idx_slice].to(device)
        node_t_to_nid = self.node_t_to_nid.to(device)

        B = idx_slice.numel()
        F = x_all.size(1)
        L = self.lookback
        hist_dtype = self._hist_dtype(x_all)
        hist = torch.zeros((B, L, F), device=device, dtype=hist_dtype)

        for l in range(L):
            # l=0 oldest, l=L-1 current
            lag = (L - 1) - l
            target_t = time_idx - lag
            valid = target_t >= 0
            if not valid.any():
                continue

            src_nodes = local_nodes[valid]
            src_times = target_t[valid]
            src_nids = node_t_to_nid[src_nodes, src_times]
            valid2 = src_nids >= 0
            if not valid2.any():
                continue

            dest_idx = torch.where(valid)[0][valid2]
            hist[dest_idx, l, :] = x_all[src_nids[valid2]].to(hist_dtype)

        return hist


class SimpleTemporalCNNClassifier(_HistoryModelBase):
    def __init__(
        self,
        in_dim: int,
        node_t_to_nid: torch.Tensor,
        local_node_idx: torch.Tensor,
        time_idx: torch.Tensor,
        lookback: int = 14,
        hist_chunk: int = 50_000,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__(
            node_t_to_nid=node_t_to_nid,
            local_node_idx=local_node_idx,
            time_idx=time_idx,
            lookback=lookback,
            hist_chunk=hist_chunk,
        )
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        del edge_index_spat, edge_index_temp
        N = x.size(0)
        device = x.device
        logits_parts = []

        for start in range(0, N, self.hist_chunk):
            end = min(start + self.hist_chunk, N)
            idx = torch.arange(start, end, device=device, dtype=torch.long)
            hist = self._build_history_slice(x, idx).to(dtype=x.dtype)  # [B, L, F]

            # Conv1d expects [B, C, L] where C=F
            h = hist.transpose(1, 2)  # [B, F, L]
            h = torch.relu(self.conv1(h))
            h = self.dropout(h)
            h = torch.relu(self.conv2(h))
            h = self.dropout(h)
            h = h.mean(dim=2)  # [B, hidden]
            logits_parts.append(self.out(h).squeeze(-1))

        return torch.cat(logits_parts, dim=0)


class SimpleTemporalLSTMClassifier(_HistoryModelBase):
    def __init__(
        self,
        in_dim: int,
        node_t_to_nid: torch.Tensor,
        local_node_idx: torch.Tensor,
        time_idx: torch.Tensor,
        lookback: int = 14,
        hist_chunk: int = 50_000,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__(
            node_t_to_nid=node_t_to_nid,
            local_node_idx=local_node_idx,
            time_idx=time_idx,
            lookback=lookback,
            hist_chunk=hist_chunk,
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_spat, edge_index_temp):
        del edge_index_spat, edge_index_temp
        N = x.size(0)
        device = x.device
        logits_parts = []

        for start in range(0, N, self.hist_chunk):
            end = min(start + self.hist_chunk, N)
            idx = torch.arange(start, end, device=device, dtype=torch.long)
            hist = self._build_history_slice(x, idx).to(dtype=x.dtype)  # [B, L, F]
            out_seq, _ = self.lstm(hist)
            h = self.dropout(out_seq[:, -1, :])  # last time step
            logits_parts.append(self.out(h).squeeze(-1))

        return torch.cat(logits_parts, dim=0)


def _model_factory(
    name: str,
    in_dim: int,
    data,
    args,
) -> Tuple[nn.Module, str]:
    name = name.lower().strip()

    if name == "simple_gcn":
        return (
            SimpleGCNClassifier(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ),
            "simple_gcn",
        )
    if name == "simple_gat":
        return (
            SimpleGATClassifier(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                heads=args.gat_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ),
            "simple_gat",
        )
    if name == "gdn":
        return (
            GDNStyleClassifier(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                recon_weight=args.gdn_recon_weight,
            ),
            "gdn",
        )
    if name == "sage":
        return (
            SAGEClassifier(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ),
            "sage",
        )
    if name == "graphwavenet":
        return (
            GraphWaveNetClassifier(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ),
            "graphwavenet",
        )
    if name == "simple_cnn":
        return (
            SimpleTemporalCNNClassifier(
                in_dim=in_dim,
                node_t_to_nid=data.node_t_to_nid,
                local_node_idx=data.local_node_idx,
                time_idx=data.time_idx,
                lookback=args.lookback,
                hist_chunk=args.hist_chunk,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            ),
            "simple_cnn",
        )
    if name == "simple_lstm":
        return (
            SimpleTemporalLSTMClassifier(
                in_dim=in_dim,
                node_t_to_nid=data.node_t_to_nid,
                local_node_idx=data.local_node_idx,
                time_idx=data.time_idx,
                lookback=args.lookback,
                hist_chunk=args.hist_chunk,
                hidden_dim=args.hidden_dim,
                num_layers=args.lstm_layers,
                dropout=args.dropout,
            ),
            "simple_lstm",
        )

    raise ValueError(f"Unknown model name: {name}")


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_builtin(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="simple_gcn,simple_gat,gdn,simple_cnn,simple_lstm",
        help="Comma-separated model names.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--gdn-recon-weight", type=float, default=0.1)

    # Sequence-model knobs
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--hist-chunk", type=int, default=50_000)
    parser.add_argument("--lstm-layers", type=int, default=1)

    # Evaluation knobs (same semantics as gtg.py)
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--c-miss", type=float, default=1.0)
    parser.add_argument("--c-false", type=float, default=0.05)
    parser.add_argument("--lowfa-target-hit", type=float, default=0.80)
    parser.add_argument("--target-fa-rate", type=float, default=-1.0)
    parser.add_argument("--n-thresholds", type=int, default=1001)

    # Data horizon knobs (same as gtg.py)
    parser.add_argument("--wind-feature", type=str, default="wind_speed")
    parser.add_argument("--gtg-years", type=float, default=12.0)
    parser.add_argument("--total-years", type=float, default=20.0)
    parser.add_argument("--no-gtg-restrict", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pos-weight-power", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=0.0)
    parser.add_argument("--out-prefix", type=str, default="other_models")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Out prefix: {args.out_prefix}")

    df = load_daily_csv(args.csv)
    print("Building graph WITHOUT wind edges (self temporal edges only)...")
    data = build_space_time_graph(
        df,
        use_wind_edges=False,
        wind_feature_name=args.wind_feature,
        restrict_to_gtg_horizon=(not args.no_gtg_restrict),
        gtg_years=args.gtg_years,
        total_years=args.total_years,
    )
    in_dim = int(data.x.size(1))

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results: Dict[str, dict] = {}

    for model_name in model_names:
        print("=" * 80)
        print(f"Experiment: {model_name}")
        print("=" * 80)

        model, model_kind = _model_factory(model_name, in_dim, data, args)

        res = train_model(
            model=model,
            data=data,
            device=torch.device(args.device),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_kind=model_kind if model_kind == "gdn" else model_name,
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

        raw = res.pop("raw", None)
        if raw is not None:
            npz_path = f"{args.out_prefix}_{model_name}_rl_data.npz"
            np.savez(npz_path, **raw)
            print(f"Saved raw outputs to {npz_path}")

        all_results[model_name] = res
        print(
            f"[{model_name}] summary: "
            f"best_thr_f1={res['best_thr_f1']:.4f}, "
            f"best_thr_ew={res['best_thr_ew']:.4f}, "
            f"val_ew_hit_rate={res['val_ew_hit_rate']:.3f}, "
            f"val_ew_FA_rate={res['val_ew_fa_rate']:.5f}"
        )

    summary_path = f"{args.out_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(all_results), f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")

    print("\n=== Global summary (test split) ===")
    for model_name, res in all_results.items():
        test_stats = res["splits"]["test"]
        ew = res["early_warning_5d"]
        print(
            f"{model_name:16s} | ROC-AUC={test_stats['roc_auc']:.4f} "
            f"PR-AUC={test_stats['pr_auc']:.4f} F1={test_stats['f1']:.4f} "
            f"| hit_rate_{args.window_days}d={ew['hit_rate']:.3f} "
            f"mean_lead_{args.window_days}d={ew['mean_lead']:.2f}d "
            f"FA_rate={ew['fa_rate']:.5f}"
        )


if __name__ == "__main__":
    main()

