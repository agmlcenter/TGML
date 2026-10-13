"""
GNN-based modules used in the architecture of MP-TG models

"""

import math

import torch_geometric.nn
from torch_geometric.nn import TransformerConv
import torch


class MGraphAttentionEmbedding(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, node_feat_dim=None):
        super().__init__()
        self.time_enc = time_enc
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if node_feat_dim is not None:
            self.node_feat_proj = torch.nn.Sequential(
                torch.nn.Linear(node_feat_dim, in_channels),
                torch.nn.LayerNorm(in_channels),
                torch.nn.GELU(),
                torch.nn.Dropout(p=0.2),
            )
            self.node_gate = torch.nn.Linear(in_channels * 2, in_channels)
            self.feature_alpha = torch.nn.Parameter(torch.tensor(-1.0))
            self.feature_dropout = torch.nn.Dropout(p=0.3)
            total_in_channels = in_channels
        else:
            self.node_feat_proj = None
            self.node_gate = None
            self.feature_alpha = None
            self.feature_dropout = None
            total_in_channels = in_channels

        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            total_in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim
        )
        self.skip_proj = torch.nn.Linear(total_in_channels, out_channels)
        self.post_mlp_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(out_channels),
            torch.nn.GELU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.Dropout(p=0.1),
        )

    def forward(self, x, last_update, edge_index, t, msg, node_feat=None):
        base_x = x
        if node_feat is not None and self.node_feat_proj is not None:
            node_feat_proj = self.feature_dropout(self.node_feat_proj(node_feat))
            gate = torch.sigmoid(self.node_gate(torch.cat([base_x, node_feat_proj], dim=-1)))
            blend = gate * node_feat_proj + (1.0 - gate) * base_x
            base_x = base_x + torch.sigmoid(self.feature_alpha) * (blend - base_x)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        res = self.conv(base_x, edge_index, edge_attr)
        skip = self.skip_proj(base_x[:res.size(0)])
        return self.post_mlp_layer(res + skip)

    def post_mlp(self, x):
        self.post_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(out_channels),
            torch.nn.GELU(),
            torch.nn.Linear(out_channels, out_channels),
        )
        return self.post_mlp(x)

    def set_feature_alpha_trainable(self, trainable: bool):
        if isinstance(self.feature_alpha, torch.nn.Parameter):
            self.feature_alpha.requires_grad_(trainable)

