import timeit
import numpy as np
import torch.nn as nn
import torch
from torch_geometric.loader import TemporalDataLoader
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm
import torch
import timeit
import argparse
import matplotlib.pyplot as plt
from regularization import HindgeRegularization
from pytorchltr.loss.pairwise_lambda import LambdaNDCGLoss2
import os
import subprocess
import atexit
import wandb
from tgb.utils.utils import set_random_seed
from typing import Optional
from torch import Tensor

LOG_FEATURE_INDICES = {0, 1, 2, 3, 4, 6, 7}
Z_SCORE_EXCLUDE_INDICES = {5}
SECTOR_FEATURE_INDEX = 5
SECTOR_EMBED_DIM = 32


def normalize_node_features(
    node_feat: Tensor,
    node_feat_time: Optional[Tensor],
    base_dataset,
    temporal_data,
    train_mask: Tensor,
    device: torch.device,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    if not torch.is_tensor(node_feat):
        node_feat = torch.as_tensor(node_feat)
    node_feat = node_feat.detach().to(dtype=torch.float32, device="cpu")

    sector_tensor: Optional[Tensor] = None
    sector_time: Optional[Tensor] = None
    sector_idx = SECTOR_FEATURE_INDEX if node_feat.shape[-1] > SECTOR_FEATURE_INDEX else None

    if node_feat_time is not None:
        if not torch.is_tensor(node_feat_time):
            node_feat_time = torch.as_tensor(node_feat_time)
        node_feat_time = node_feat_time.detach().to(device="cpu", dtype=temporal_data.t.dtype)
    elif node_feat.dim() == 3:
        label_ts = getattr(base_dataset, "label_ts", None)
        if label_ts is not None:
            node_feat_time = torch.as_tensor(label_ts, dtype=temporal_data.t.dtype)
        else:
            node_feat_time = torch.arange(node_feat.size(0), dtype=temporal_data.t.dtype)

    feat_dim = node_feat.shape[-1]
    log_indices = sorted(idx for idx in LOG_FEATURE_INDICES if idx < feat_dim)
    standardise_indices = sorted(idx for idx in range(feat_dim) if idx not in Z_SCORE_EXCLUDE_INDICES)

    if node_feat.dim() == 3:
        time_reference = node_feat_time
        if time_reference is None or time_reference.numel() != node_feat.size(0):
            time_reference = torch.arange(node_feat.size(0), dtype=temporal_data.t.dtype)

        train_event_times = temporal_data.t[train_mask].detach().cpu()
        if train_event_times.numel() > 0:
            cutoff = train_event_times.max()
            time_reference_eval = time_reference.to(dtype=cutoff.dtype)
            train_time_mask = time_reference_eval <= cutoff
        else:
            train_time_mask = torch.ones(node_feat.size(0), dtype=torch.bool)
        if not train_time_mask.any():
            train_time_mask = torch.ones(node_feat.size(0), dtype=torch.bool)

        if log_indices:
            idx_tensor = torch.tensor(log_indices, dtype=torch.long)
            node_feat[..., idx_tensor] = torch.log1p(torch.clamp(node_feat[..., idx_tensor], min=0.0))

        train_slice = node_feat[train_time_mask]
        mean_vec = torch.zeros(feat_dim, dtype=torch.float32)
        std_vec = torch.ones(feat_dim, dtype=torch.float32)
        if standardise_indices:
            feature_mean = train_slice.mean(dim=(0, 1))
            feature_std = train_slice.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
            idx_tensor = torch.tensor(standardise_indices, dtype=torch.long)
            mean_vec[idx_tensor] = feature_mean[idx_tensor]
            std_vec[idx_tensor] = feature_std[idx_tensor]
        node_feat = (node_feat - mean_vec.view(1, 1, -1)) / std_vec.view(1, 1, -1)
        node_feat_time = time_reference
        sector_time = node_feat_time

    elif node_feat.dim() == 2:
        if log_indices:
            idx_tensor = torch.tensor(log_indices, dtype=torch.long)
            node_feat[..., idx_tensor] = torch.log1p(torch.clamp(node_feat[..., idx_tensor], min=0.0))
        mean_vec = torch.zeros(feat_dim, dtype=torch.float32)
        std_vec = torch.ones(feat_dim, dtype=torch.float32)
        if standardise_indices:
            feature_mean = node_feat.mean(dim=0)
            feature_std = node_feat.std(dim=0, unbiased=False).clamp_min(1e-6)
            idx_tensor = torch.tensor(standardise_indices, dtype=torch.long)
            mean_vec[idx_tensor] = feature_mean[idx_tensor]
            std_vec[idx_tensor] = feature_std[idx_tensor]
        node_feat = (node_feat - mean_vec.view(1, -1)) / std_vec.view(1, -1)
        node_feat_time = None
        sector_time = None
    else:
        raise ValueError(f"Unsupported node feature shape: {node_feat.shape}")

    if sector_idx is not None:
        if node_feat.dim() == 3:
            sector_tensor = node_feat[..., sector_idx].round().clamp_min(0).to(torch.long)
            node_feat = torch.cat([node_feat[..., :sector_idx], node_feat[..., sector_idx + 1 :]], dim=-1)
        else:
            sector_tensor = node_feat[..., sector_idx].round().clamp_min(0).to(torch.long)
            node_feat = torch.cat([node_feat[..., :sector_idx], node_feat[..., sector_idx + 1 :]], dim=-1)
        if sector_tensor.numel() > 0:
            sector_min = sector_tensor.min()
            sector_tensor = (sector_tensor - sector_min).clamp_min(0)
    node_feat = node_feat.to(device=device)
    if node_feat_time is not None:
        node_feat_time = node_feat_time.to(device=device, dtype=temporal_data.t.dtype)
    if sector_tensor is not None:
        sector_tensor = sector_tensor.to(device=device, dtype=torch.long)
        if sector_time is not None:
            sector_time = sector_time.to(device=device, dtype=temporal_data.t.dtype)
    return node_feat, node_feat_time, sector_tensor, sector_time


class SimpleRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool):
        super(SimpleRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.linx = nn.Linear(input_size, 1)
        self.linh = nn.Linear(hidden_size, 1)

        self.ling = nn.Linear(input_size, 1)
        self.linxg = nn.Linear(input_size, 1)
        self.linhg = nn.Linear(hidden_size, 1)

        self.s1 = torch.nn.Sigmoid()
        self.s2 = torch.nn.Sigmoid()

    def forward(self, h: torch.tensor, x: torch.tensor, g: torch.tensor):
        z1 = self.s1(self.linx(x) + self.linh(h))
        h_tild = z1 * h + (1 - z1) * x
        z2 = self.s2(self.ling(g) + self.linxg(x) + self.linhg(h))
        h_new = z2 * h_tild + (1 - z2) * x
        return h_new, h_tild


class LearnableMovingAverage(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_class: int,
        is_multi: bool = False,
        node_features: Optional[torch.Tensor] = None,
        node_feat_time: Optional[torch.Tensor] = None,
        node_sectors: Optional[torch.Tensor] = None,
        node_sector_time: Optional[torch.Tensor] = None,
    ):
        super(LearnableMovingAverage, self).__init__()
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.is_multi = is_multi
        self.semi_labels = torch.zeros(num_nodes, num_class, device=device).float()
        self.node_history = torch.zeros((num_nodes, num_class), device=device).float()
        self.node_prev_label = torch.zeros((num_nodes, num_class), device=device).float()
        self.lstm = SimpleRNNCell(input_size=num_classes, hidden_size=num_classes, bias=False)
        self.prev_global_label = torch.zeros((1, num_class), device=device).float()

        if node_features is None:
            raise ValueError("NAVIS requires `data.node_feat` to provide temporal node features.")
        node_features = node_features.to(device=device, dtype=torch.float32)
        self.register_buffer("node_features", node_features)
        if node_feat_time is not None:
            self.register_buffer("node_feat_time", node_feat_time.to(device=device))
        else:
            self.node_feat_time = None

        self.sector_emb = None
        if node_sectors is not None and node_sectors.numel() > 0:
            node_sectors = node_sectors.to(device=device, dtype=torch.long)
            self.register_buffer("node_sectors", node_sectors)
            if node_sector_time is not None:
                self.register_buffer("node_sector_time", node_sector_time.to(device=device))
            else:
                self.node_sector_time = None
            num_sectors = int(node_sectors.max().item()) + 1
            self.sector_emb = nn.Embedding(num_embeddings=num_sectors, embedding_dim=SECTOR_EMBED_DIM)
        else:
            self.node_sectors = None
            self.node_sector_time = None

        feature_dim = node_features.shape[-1]
        if self.sector_emb is not None:
            feature_dim += self.sector_emb.embedding_dim
        self.feature_proj = nn.Linear(feature_dim, num_class)

    def get_memory_and_update(self, node_ids: np.array, labels: np.array):
        """
        This function returns the memory of the nodes and updates the memory with the new labels, one by one.

        :param node_ids: np.array, shape (batch_size,)
        :param labels: np.array, shape (batch_size, num_class)

        :return: np.array, shape (batch_size, window, num_class)
        """
        assert len(node_ids) == len(set(node_ids)), "node_ids should be unique"

        if self.is_multi:
            pred, state = self.lstm(self.node_history[node_ids].clone(),labels,labels)
            self.node_history[node_ids] = state
        else:
            gs = labels.to(device).clone()[0:labels.shape[0] - 1]
            gs = torch.cat([self.prev_global_label, gs], dim=0)
            self.prev_global_label = labels[labels.shape[0] - 1].reshape(1, -1).to(device)

            pred, state = self.lstm(self.node_history[node_ids].clone(), self.node_prev_label[node_ids].clone(), gs)
            self.node_prev_label[node_ids] = labels
            self.node_history[node_ids] = state

        return pred

    def forward(self, node_ids: np.array, timestamps: np.array, labels: np.array):
        """
        This function predicts the labels of the nodes.

        :param node_ids: np.array, shape (batch_size,)
        :param timestamps: np.array, shape (batch_size,)
        :param labels: np.array, shape (batch_size, num_class)

        :return: np.array, shape (batch_size, num_class)
        """
        if self.is_multi:
            semi_labels = get_semilabels(node_ids)
        else:
            semi_labels = labels
        memories = self.get_memory_and_update(node_ids, semi_labels)
        feature_embedding = self._encode_node_features(node_ids, timestamps)

        return memories + feature_embedding

    def reset(self):
        self.node_history = torch.zeros((self.num_nodes, self.num_class), device=device).float()
        self.node_prev_label = torch.zeros((self.num_nodes, self.num_class), device=device).float()
        self.semi_labels = torch.zeros((self.num_nodes, self.num_class), device=device).float()
        self.prev_global_label = torch.zeros((1, self.num_class), device=device).float()
        # node_features and feature_proj parameters remain unchanged across resets

    def _encode_node_features(self, node_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        feats = self._gather_node_features(node_ids, timestamps)
        if self.sector_emb is not None and self.node_sectors is not None:
            sector_ids = self._gather_sector_features(node_ids, timestamps)
            sector_emb = self.sector_emb(sector_ids)
            feats = torch.cat([feats, sector_emb], dim=-1)
        return self.feature_proj(feats)

    def _gather_node_features(self, node_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.to(dtype=torch.long)
        if self.node_features.dim() == 2:
            return self.node_features[node_ids]

        timestamps = timestamps.to(self.node_features.device)
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)

        if self.node_feat_time is not None:
            ts = timestamps.to(self.node_feat_time.dtype)
            time_indices = torch.searchsorted(self.node_feat_time, ts)
        else:
            time_indices = timestamps.to(torch.long)
        max_index = self.node_features.shape[0] - 1
        time_indices = time_indices.clamp_(0, max_index)
        return self.node_features[time_indices, node_ids]

    def _gather_sector_features(self, node_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        if self.node_sectors.dim() == 1:
            return self.node_sectors[node_ids.to(dtype=torch.long)]
        timestamps = timestamps.to(self.node_sectors.device)
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        if self.node_sector_time is not None:
            ts = timestamps.to(self.node_sector_time.dtype)
            time_indices = torch.searchsorted(self.node_sector_time, ts)
        else:
            time_indices = timestamps.to(torch.long)
        max_index = self.node_sectors.shape[0] - 1
        time_indices = time_indices.clamp_(0, max_index)
        return self.node_sectors[time_indices, node_ids.to(dtype=torch.long)]


parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='random seed to use')
parser.parse_args()
args = parser.parse_args()
# setting random seed
seed = int(args.seed)  # 1,2,3,4,5
seed = 2
print("setting random seed to be", seed)
torch.manual_seed(seed)
set_random_seed(seed)

# hyperparameters
batch_size = 200
lr = 0.0001
epochs = 500
max_classes = 20
IS_MULTI = False # whether to use full CTDG instead of previous ground truth labels in the prediction
DELTA = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = "tgbn-13f"
dataset = PyGNodePropPredDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
temporal_data = dataset.get_TemporalData()
num_nodes = temporal_data.num_nodes

eval_metric = dataset.eval_metric
num_classes = dataset.num_classes

base_dataset = getattr(dataset, "dataset", None)
if base_dataset is None or not hasattr(base_dataset, "node_feat") or base_dataset.node_feat is None:
    raise ValueError("Node features must be provided via NodePropPredDataset.node_feat for NAVIS.")
node_feat_array = base_dataset.node_feat
if isinstance(node_feat_array, np.ndarray):
    node_feat_tensor = torch.from_numpy(node_feat_array)
else:
    node_feat_tensor = node_feat_array
node_feat_time = getattr(base_dataset, "node_feat_time", None)
node_feat_tensor, node_feat_time, node_sector_tensor, node_sector_time = normalize_node_features(
    node_feat_tensor,
    node_feat_time,
    base_dataset,
    temporal_data,
    train_mask,
    device,
)

data = temporal_data.to(device)
evaluator = Evaluator(name=name)

run_name = f"navis_dataset_{name}_bs_{batch_size}_lr_{lr}_epochs_{epochs}_seed_{seed}"
wandb_kwargs = {
    "project": "navis",
    "entity": "emad-izadifar-amirkabir-university-of-technology",
    "name": run_name,
    "config": {
        "dataset": name,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "max_classes": max_classes,
        "is_multi": IS_MULTI,
        "delta": DELTA,
    },
}
wandb_desired_mode = os.getenv("WANDB_MODE", "online")
actual_wandb_mode = wandb_desired_mode
wandb_run = None
try:
    wandb_run = wandb.init(mode=wandb_desired_mode, **wandb_kwargs)
    actual_wandb_mode = getattr(getattr(wandb_run, "settings", None), "mode", wandb_desired_mode)
except Exception as wandb_exc:
    if wandb_desired_mode.lower() != "offline":
        print(f"[wandb] Init failed in mode '{wandb_desired_mode}'. Falling back to offline. Error: {wandb_exc}")
        wandb_run = wandb.init(mode="offline", reinit=True, **wandb_kwargs)
        actual_wandb_mode = "offline"
    else:
        raise
wandb_synced = False

def _sync_wandb_run():
    global wandb_synced
    if wandb_synced:
        return
    run = wandb_run if wandb_run is not None else wandb.run
    if run is None:
        return
    run_dir = getattr(run, "dir", None)
    try:
        wandb.finish(exit_code=0)
    finally:
        if actual_wandb_mode.lower() == "offline" and run_dir and os.path.isdir(run_dir):
            subprocess.run(["wandb", "sync", run_dir], check=False)
    wandb_synced = True

atexit.register(_sync_wandb_run)


def process_edges(src, dst, t, msg):
    msg = msg[:, 0:1].abs()  # take only the first column of the message
    node_pred.semi_labels.index_put_((src, dst), msg.reshape(-1), accumulate=True)


def get_semilabels(src):
    with torch.no_grad():

        sl = node_pred.semi_labels

        # Make sure index dtype/device match
        src = src.to(dtype=torch.long, device=sl.device)
        logits = sl[src].clone()

        # scale logits by the sum of logits for each node
        scale = logits.sum(dim=1, keepdim=True)
        # Avoid division by zero by using a safe denominator
        logits = torch.where(scale > 0, logits / scale, logits)
        node_pred.semi_labels[src] = 0

    return logits


train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

node_pred = LearnableMovingAverage(
    num_nodes,
    num_classes,
    is_multi=IS_MULTI,
    node_features=node_feat_tensor,
    node_feat_time=node_feat_time,
    node_sectors=node_sector_tensor,
    node_sector_time=node_sector_time,
).to(device)

optimizer = torch.optim.Adam(node_pred.parameters(), lr=lr)
criterion = LambdaNDCGLoss2()
regularization = HindgeRegularization(top_k=max_classes, delta=DELTA)


def plot_curve(scores, out_name):
    plt.plot(scores, color="#e34a33")
    plt.ylabel("score")
    plt.savefig(out_name + ".pdf")
    plt.close()


def prepare_logits(preds: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    if preds.size(1) > max_classes:
        _, p_idx = torch.topk(preds, max_classes)
        _, l_idx = torch.topk(labels, max_classes)
        top_idxs = torch.unique(torch.cat([p_idx, l_idx], dim=1), dim=1)
        preds = torch.gather(preds, dim=1, index=top_idxs)
        labels = torch.gather(labels, dim=1, index=top_idxs)
    return preds, labels


def train(epoch: int):
    node_pred.train()
    node_pred.reset()  # Start with a fresh memory.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    total_score = 0
    loss_count = 0
    num_label_ts = 0
    max = 0
    min = 9999

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        maxs = msg.max()
        mins = msg.min()
        max = maxs if maxs > max else max
        min = mins if mins < min else min
        query_t = batch.t[-1]
        # check if this batch moves to the next day
        if query_t > label_t:
            # find the node labels from the past day
            label_tuple = dataset.get_node_label(query_t)
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            node_ids = label_srcs
            timestamps = label_ts
            np_pred = []
            np_true = []
            preds = torch.empty((0, num_classes), device=device)
            optimizer.zero_grad()
            seen_node_ids = set()
            curr_node_ids = []
            curr_timestamps = []
            curr_labels = []
            for node_id, timestamp, label in zip(node_ids, timestamps, labels):
                if node_id not in seen_node_ids:
                    seen_node_ids.add(node_id)
                    curr_node_ids.append(node_id)
                    curr_timestamps.append(timestamp)
                    curr_labels.append(label)
                else:
                    curr_node_ids = torch.tensor(curr_node_ids, device=device)
                    curr_timestamps = torch.tensor(curr_timestamps, device=device)
                    curr_labels = torch.tensor(np.stack(curr_labels), device=device)
                    pred = node_pred(curr_node_ids, curr_timestamps, curr_labels)
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    preds = torch.cat([preds, pred])
                    curr_node_ids = [node_id]
                    curr_timestamps = [timestamp]
                    curr_labels = [label]
                    seen_node_ids = {node_id}

            if len(curr_node_ids) > 0:
                curr_node_ids = torch.tensor(curr_node_ids, device=device)
                curr_timestamps = torch.tensor(curr_timestamps, device=device)
                curr_labels = torch.tensor(np.stack(curr_labels), device=device)
                pred = node_pred(curr_node_ids, curr_timestamps, curr_labels)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                preds = torch.cat([preds, pred])

            labels = labels.to(device)
            preds, labels = prepare_logits(preds, labels)

            loss = criterion(preds, labels,
                            torch.count_nonzero(labels, dim=1).clamp(max=max_classes).to(device)).mean() + \
                  regularization(preds, labels.to(device))
            total_loss += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
            })
            np_pred.append(preds.cpu().detach().numpy())
            np_true.append(labels.cpu().detach().numpy())
            with torch.no_grad():
                node_pred.node_history = node_pred.node_history.detach()
                node_pred.node_prev_label = node_pred.node_prev_label.detach()
            input_dict = {
                "y_true": np.array(np_true).reshape(-1, preds.shape[1]),
                "y_pred": np.array(np_pred).reshape(-1, preds.shape[1]),
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_label_ts += 1

            process_edges(src, dst, t, msg)
            continue  # skip the outer scope until labels exist

        process_edges(src, dst, t, msg)

    metric_dict = {
        "ce": total_loss / loss_count,
    }
    metric_dict[eval_metric] = total_score / num_label_ts
    return metric_dict


@torch.no_grad()
def test(loader):
    node_pred.eval()

    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0
    total_loss = 0.0

    for batch in tqdm(loader):
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            # Reset edges to be the edges from tomorrow so they can be used later
            # Process all edges that are still in the past day
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
            )
            # Reset edges to be the edges from tomorrow so they can be used later
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            """
            modified for node property prediction
            1. sample neighbors from the neighbor loader for all nodes to be predicted
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            node_ids = label_srcs
            timestamps = label_ts
            timestamps = timestamps.to(device)
            labels = labels.to(device)
            pred = node_pred(node_ids, timestamps, labels)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            preds, labels = prepare_logits(pred, labels)
            loss = criterion(
                preds,
                labels,
                torch.count_nonzero(labels, dim=1).clamp(max=max_classes).to(device),
            ).mean() + regularization(preds, labels.to(device))
            total_loss += float(loss)

            np_pred = preds.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_label_ts += 1
        process_edges(src, dst, t, msg)

    if num_label_ts == 0:
        return {"ce": 0.0, eval_metric: 0.0}
    metric_dict = {
        "ce": total_loss / num_label_ts,
        eval_metric: total_score / num_label_ts,
    }
    return metric_dict


train_curve = []
val_curve = []
test_curve = []
max_val_score = 0  # find the best test score based on validation score
best_test_idx = 0
train_loss = []
val_ncdg = []
val_loss_curve = []
test_loss_curve = []
for epoch in range(1, epochs + 1):
    start_time = timeit.default_timer()
    train_dict = train(epoch)
    print("------------------------------------")
    print(f"training Epoch: {epoch:02d}")
    print(train_dict)
    train_curve.append(train_dict[eval_metric])
    wandb.log({
        "train/train_loss": train_dict["ce"],
        f"train/{eval_metric}": train_dict[eval_metric],
        "train/epoch": epoch,
    })
    print("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))

    train_loss.append(train_dict['ce'])

    start_time = timeit.default_timer()
    val_dict = test(val_loader)
    print(val_dict)

    val_ncdg.append(val_dict[eval_metric])
    val_loss_curve.append(val_dict["ce"])

    with open(f"{name}_logs.txt", "a") as f:
        f.write(f"{epoch},{train_dict['ce']},{train_dict[eval_metric]},{val_dict[eval_metric]}\n")
    val_curve.append(val_dict[eval_metric])
    if (val_dict[eval_metric] > max_val_score):
        max_val_score = val_dict[eval_metric]
        best_test_idx = epoch - 1
    print("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    test_dict = test(test_loader)
    print(test_dict)
    test_curve.append(test_dict[eval_metric])
    wandb.log({
        f"test/{eval_metric}": test_dict[eval_metric],
        "test/test_loss": test_dict["ce"],
        "test/epoch": epoch,
    })
    val_log_payload = {
        f"val/{eval_metric}": val_dict[eval_metric],
        "val/val_loss": val_dict["ce"],
        "val/epoch": epoch,
    }
    wandb.log(val_log_payload)
    print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    print("------------------------------------")
    dataset.reset_label_time()
    test_loss_curve.append(test_dict["ce"])

# # code for plotting
plot_curve(train_curve, f"{run_name}_train_curve")
plot_curve(val_curve, f"{run_name}_val_curve")
plot_curve(test_curve, f"{run_name}_test_curve")
plot_curve(train_loss, f"{run_name}_train_loss")
plot_curve(val_loss_curve, f"{run_name}_val_loss")
plot_curve(test_loss_curve, f"{run_name}_test_loss")

max_test_score = test_curve[best_test_idx]
print("------------------------------------")
print("------------------------------------")
print("best val score: ", max_val_score)
print("best validation epoch   : ", best_test_idx + 1)
print("best test score: ", max_test_score)

# save train_loss, val_ncdg
with open("log_our_loss.csv", "w") as f:
    f.write("epoch,train_loss,val_ncdg\n")
    for i in range(len(train_loss)):
        f.write(f"{i+1},{train_loss[i]},{val_ncdg[i]}\n")

wandb_summary = wandb_run.summary if wandb_run is not None else wandb.run.summary
wandb_summary["best_val"] = max_val_score
wandb_summary["best_test"] = max_test_score
_sync_wandb_run()