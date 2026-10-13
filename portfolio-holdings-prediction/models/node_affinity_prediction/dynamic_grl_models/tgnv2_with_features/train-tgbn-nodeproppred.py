import timeit
import torch
import wandb
import os
import math
import utils
import numpy as np
import subprocess
import atexit

from tqdm import tqdm
import torch.nn as nn
from models.mtgn import MTGNMemory, LastAggregator, LastNeighborLoader
from models.embmodule import MGraphAttentionEmbedding
from models.msgmodule import EncodeIndexModule
from models.decoder import NodePredictor
from logger.logger import Logger

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
)

from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset, TemporalData
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed

from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from typing import Optional

LOG_FEATURE_INDICES = {0, 1, 2, 3, 4, 6, 7}
Z_SCORE_EXCLUDE_INDICES = {5}
SECTOR_FEATURE_INDEX = 5
SECTOR_EMBED_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_temporal_node_features(
        node_feat: Tensor,
        node_feat_time: Optional[Tensor],
        base_dataset,
        temporal_data: TemporalData,
        train_mask) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if not torch.is_tensor(node_feat):
        node_feat = torch.as_tensor(node_feat)
    node_feat = node_feat.detach().to(dtype=torch.float32, device="cpu")

    if node_feat_time is not None:
        if not torch.is_tensor(node_feat_time):
            node_feat_time = torch.as_tensor(node_feat_time)
        node_feat_time = node_feat_time.detach().to(dtype=temporal_data.t.dtype)
    elif node_feat.dim() == 3:
        label_ts = getattr(base_dataset, "label_ts", None)
        if label_ts is not None:
            node_feat_time = torch.as_tensor(label_ts, dtype=temporal_data.t.dtype)
        else:
            node_feat_time = torch.arange(node_feat.size(0), dtype=temporal_data.t.dtype)

    feat_dim = node_feat.shape[-1]
    log_indices = sorted(idx for idx in LOG_FEATURE_INDICES if idx < feat_dim)
    standardise_indices = sorted(idx for idx in range(feat_dim) if idx not in Z_SCORE_EXCLUDE_INDICES)
    sector_tensor = None
    sector_idx = SECTOR_FEATURE_INDEX if feat_dim > SECTOR_FEATURE_INDEX else None

    if node_feat.dim() == 3:
        time_reference = node_feat_time
        if time_reference is None or time_reference.numel() != node_feat.size(0):
            time_reference = torch.arange(node_feat.size(0), dtype=temporal_data.t.dtype)
        if not torch.is_tensor(train_mask):
            mask_tensor = torch.as_tensor(train_mask, dtype=torch.bool, device=temporal_data.t.device)
        else:
            mask_tensor = train_mask.to(dtype=torch.bool, device=temporal_data.t.device)
        train_event_times = temporal_data.t[mask_tensor].detach().cpu()
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
    else:
        raise ValueError(f"Unsupported node feature shape: {node_feat.shape}")

    if sector_idx is not None:
        if node_feat.dim() == 3:
            sector_tensor = node_feat[..., sector_idx].round().clamp_min(0).to(torch.long)
            node_feat = torch.cat([node_feat[..., :sector_idx], node_feat[..., sector_idx + 1:]], dim=-1)
        else:
            sector_tensor = node_feat[..., sector_idx].round().clamp_min(0).to(torch.long)
            node_feat = torch.cat([node_feat[..., :sector_idx], node_feat[..., sector_idx + 1:]], dim=-1)
        if sector_tensor.numel() > 0:
            sector_tensor = (sector_tensor - sector_tensor.min()).clamp_min(0)
    return node_feat, node_feat_time, sector_tensor


def gather_temporal_node_features(
        node_feats: Tensor,
        node_sectors: Optional[Tensor],
        sector_emb: Optional[nn.Embedding],
        time_idx: torch.Tensor,
        node_ids: torch.Tensor) -> Tensor:
    if isinstance(time_idx, torch.Tensor):
        time_idx = int(time_idx.item())
    node_ids = node_ids.to(dtype=torch.long)
    feats = node_feats[time_idx, node_ids, :]
    if node_sectors is not None and sector_emb is not None:
        if node_sectors.dim() == 3:
            sector_ids = node_sectors[time_idx, node_ids]
        elif node_sectors.dim() == 2:
            sector_ids = node_sectors[time_idx, node_ids]
        else:
            sector_ids = node_sectors[node_ids]
        feats = torch.cat([feats, sector_emb(sector_ids)], dim=-1)
    return feats


# Starting Point:
# https://github.com/shenyangHuang/TGB/blob/main/examples/nodeproppred/tgbn-genre/tgn.py

def process_edges(memory, src, dst, t, msg, neighbor_loader):
    if src.nelement() > 0:
        # msg = msg.to(torch.float32)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def train(
        epoch,
        memory,
        gnn,
        node_pred,
        lr_scheduler: LRScheduler,
        dataset: PyGNodePropPredDataset,
        data: TemporalData,
        evaluator: Evaluator,
        neighbor_loader,
        train_loader,
        optimizer,
        assoc,
        use_gnn=True,
        node_feats=None,
        node_sectors=None,
        sector_emb=None):
    eval_metric = dataset.eval_metric

    memory.train()
    gnn.train()
    node_pred.train()

    feature_alpha_warmup = 5
    if hasattr(gnn, "set_feature_alpha_trainable"):
        gnn.set_feature_alpha_trainable(epoch > feature_alpha_warmup)

    criterion = torch.nn.CrossEntropyLoss()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0

    train_loader_length = len(train_loader)
    count = 0
    if sector_emb is not None:
        sector_emb.train()
    for batch in tqdm(train_loader):
        count += 1
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

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
                memory,
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                neighbor_loader
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
            2. extract memory
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)
            if use_gnn:
                T = node_feats.size(0)
                sorted_timestamps = torch.arange(T, device=device)
                q_t_scalar = query_t.max()  # a scalar representative for this batch
                current_time_idx = torch.searchsorted(sorted_timestamps, q_t_scalar).clamp(0, T - 1)

                node_feat_batch = gather_temporal_node_features(
                    node_feats,
                    node_sectors,
                    sector_emb,
                    current_time_idx,
                    n_id_neighbors,
                )

                z = gnn(
                    z,
                    last_update,
                    mem_edge_index,
                    data.t[e_id].to(device),
                    data.msg[e_id].to(device),
                    node_feat_batch # Correctly indexed node features
                )

            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)

            loss = criterion(pred, labels.to(device))
            np_pred = pred.cpu().detach().numpy()
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

            loss.backward()
            optimizer.step()
            total_loss += float(loss)

            metrics = {
                "train/loss": total_loss / num_label_ts,
                "train/epoch": count / train_loader_length + epoch,
                f"train/{eval_metric}": total_score / num_label_ts,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            wandb.log(metrics)

        # Update memory and neighbor loader with ground-truth state.
        process_edges(memory, src, dst, t, msg, neighbor_loader)
        memory.detach()

    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    metrics = {
        "train/train_loss": total_loss / num_label_ts,
        "train/epoch": count / train_loader_length + epoch,
        f"train/{eval_metric}": total_score / num_label_ts,
        "train/lr": current_lr,
    }
    wandb.log(metrics)
    return metrics


@torch.no_grad()
def test(
         memory,
         gnn,
         node_pred,
         dataset: PyGNodePropPredDataset,
         data: TemporalData,
         evaluator: Evaluator,
         neighbor_loader,
         loader,
         assoc,
         split: str,
         use_gnn=True,
         node_feats=None,
         node_sectors=None,
         sector_emb=None,
         epoch=None):

    eval_metric = dataset.eval_metric

    memory.eval()
    gnn.eval()
    node_pred.eval()
    total_score = 0
    total_loss = 0.0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    criterion = torch.nn.CrossEntropyLoss()

    if sector_emb is not None:
        sector_emb.eval()
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
            process_edges(
                memory,
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
                neighbor_loader
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
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z,  last_update = memory(n_id_neighbors)
            if use_gnn:
                # Map query_t to valid indices in node_feats
                T = node_feats.size(0)
                sorted_timestamps = torch.arange(T, device=device)
                q_t_scalar = query_t.max()  # a scalar representative for this batch
                current_time_idx = torch.searchsorted(sorted_timestamps, q_t_scalar).clamp(0, T - 1)

                # Extract node features for the current time and batch of nodes
                node_feat_batch = gather_temporal_node_features(
                    node_feats,
                    node_sectors,
                    sector_emb,
                    current_time_idx,
                    n_id_neighbors,
                )

                z = gnn(
                    z,
                    last_update,
                    mem_edge_index,
                    data.t[e_id].to(device),
                    data.msg[e_id].to(device),
                    node_feat_batch  # Correctly indexed node features
                )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = node_pred(z)
            loss = criterion(pred, labels.to(device))
            total_loss += float(loss)
            np_pred = pred.cpu().detach().numpy()
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

        process_edges(memory, src, dst, t, msg, neighbor_loader)

    if num_label_ts == 0:
        metric_dict = {
            f"{split}/{eval_metric}": 0.0,
            f"{split}/{split}_loss": 0.0,
        }
    else:
        metric_dict = {
            f"{split}/{eval_metric}": total_score / num_label_ts,
            f"{split}/{split}_loss": total_loss / num_label_ts,
        }
    if epoch is not None:
        metric_dict[f"{split}/epoch"] = epoch
    wandb.log(metric_dict)
    return metric_dict


def main(args):
    # setting random seed
    seed = int(args.seed)  # 1,2,3,4,5
    torch.manual_seed(seed)
    set_random_seed(seed)

    LOG_DIR = 'logs/tidy/'
    os.makedirs(LOG_DIR, exist_ok=True)

    name = args.dataset
    dataset = PyGNodePropPredDataset(name=name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    num_classes = dataset.num_classes
    data = dataset.get_TemporalData()
    data = data.to(device)
    base_dataset = dataset.dataset
    node_feat_array = getattr(base_dataset, "node_feat", None)
    if node_feat_array is None:
        node_feat_array = getattr(base_dataset, "_node_feat", None)
    if node_feat_array is None:
        raise ValueError("Underlying dataset must expose node_feat for feature normalization.")
    node_feat_time = getattr(base_dataset, "node_feat_time", None)
    if node_feat_time is None:
        node_feat_time = getattr(base_dataset, "_node_feat_time", None)
    node_feat_tensor, node_feat_time, node_sector_tensor = normalize_temporal_node_features(
        node_feat_array,
        node_feat_time,
        base_dataset,
        data,
        train_mask,
    )
    node_feats = node_feat_tensor.to(device)
    node_sectors = node_sector_tensor.to(device) if node_sector_tensor is not None else None
    sector_emb = None
    sector_feat_dim = 0
    if node_sectors is not None and node_sectors.numel() > 0:
        num_sectors = int(node_sectors.max().item()) + 1
        sector_emb = nn.Embedding(num_embeddings=num_sectors, embedding_dim=SECTOR_EMBED_DIM).to(device)
        sector_feat_dim = SECTOR_EMBED_DIM

    evaluator = Evaluator(name=name)

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    all_hidden_dims = args.global_hidden_dims
    last_neighbour = args.num_last_neighbours

    train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
    test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

    use_gnn=True

    memory_dim = all_hidden_dims
    idx_dim = all_hidden_dims if args.use_tgnv2 else -1
    embedding_dim = all_hidden_dims
    time_dim = all_hidden_dims
    raw_msg_dim = data.msg.size(-1)

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=last_neighbour, device=device)
    msg_module = EncodeIndexModule(idx_dim, data.msg.size(-1), memory_dim, time_dim) if \
        args.use_tgnv2 else IdentityMessage(data.msg.size(-1), memory_dim, time_dim)

    aggregator_module = LastAggregator(msg_module.out_channels)

    memory = MTGNMemory(
        data.num_nodes,
        raw_msg_dim,
        memory_dim,
        time_dim,
        idx_dim,
        message_module=msg_module,
        aggregator_module=aggregator_module,
    ).to(device)

    gnn = (
        MGraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
            node_feat_dim=node_feats.size(2) + sector_feat_dim
        )
        .to(device)
        .float()
    )

    node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)

    param_list = list(memory.parameters()) + list(gnn.parameters()) + list(node_pred.parameters())
    if sector_emb is not None:
        param_list += list(sector_emb.parameters())
    optimizer = torch.optim.Adam(param_list, lr=lr)

    if args.learning_scheduler == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        scheduler_desc = ''
    elif args.learning_scheduler == 'cosine_annealing':
        ratio = args.cosine_annealing_ratio
        t_max = int(math.ceil(args.epochs / ratio))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        scheduler_desc = '_ratio_{}'.format(ratio)
    elif args.learning_scheduler == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma)
        scheduler_desc = '_gamma_{}_step_size_{}'.format(args.step_lr_gamma, args.step_lr_step_size)
    else:
        raise ValueError

    model_name = 'tgnv2' if args.use_tgnv2 else 'tgn'
    run_name = 'scheduler_{}_{}_dataset_{}_bs_{}_lr_{}_epochs_{}_last_neighbour_{}_global_dims_{}_seed_{}'.format(
        args.learning_scheduler + scheduler_desc, model_name, name, batch_size, lr, epochs, last_neighbour, all_hidden_dims, seed)

    wandb_kwargs = {
        "project": "tgnv2",
        "entity": "emad-izadifar-amirkabir-university-of-technology",
        "name": run_name,
        "config": vars(args),
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
        nonlocal wandb_synced
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

    logger = Logger(os.path.join(LOG_DIR, run_name))
    # lr_scheduler = StepLR(optimizer, 250, gamma=0.5)
    # lr_scheduler = None

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    max_val_score = 0  # find the best test score based on validation score
    best_test_idx = 0

    test_ndcgs = []

    eval_metric = dataset.eval_metric
    for epoch in range(1, epochs + 1):
        start_time = timeit.default_timer()
        train_dict = train(
            epoch=epoch,
            memory=memory,
            gnn=gnn,
            node_pred=node_pred,
            lr_scheduler=lr_scheduler,
            dataset=dataset,
            data=data,
            evaluator=evaluator,
            neighbor_loader=neighbor_loader,
            train_loader=train_loader,
            optimizer=optimizer,
            assoc=assoc,
            use_gnn=use_gnn,
            node_feats=node_feats,
            node_sectors=node_sectors,
            sector_emb=sector_emb,
         )
        logger.log_and_write("------------------------------------")
        logger.log_and_write(f"training Epoch: {epoch:02d}")
        logger.log_and_write(train_dict)
        logger.log_and_write("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        val_dict = test(
            memory=memory,
            gnn=gnn,
            node_pred=node_pred,
            dataset=dataset,
            data=data,
            evaluator=evaluator,
            neighbor_loader=neighbor_loader,
            assoc=assoc,
            loader=val_loader,
            use_gnn=use_gnn,
            split='val',
            node_feats=node_feats,
            node_sectors=node_sectors,
            sector_emb=sector_emb,
             epoch=epoch,
         )

        logger.log_and_write(val_dict)
        val_ndcg = val_dict[f'val/{eval_metric}']
        if val_ndcg > max_val_score:
            max_val_score = val_ndcg
            best_test_idx = epoch - 1
        logger.log_and_write("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        test_dict = test(
            memory=memory,
            gnn=gnn,
            node_pred=node_pred,
            dataset=dataset,
            data=data,
            evaluator=evaluator,
            neighbor_loader=neighbor_loader,
            assoc=assoc,
            loader=test_loader,
            use_gnn=use_gnn,
            split='test',
            node_feats=node_feats,
            node_sectors=node_sectors,
            sector_emb=sector_emb,
             epoch=epoch,
         )
        test_ndcgs.append(test_dict[f'test/{eval_metric}'])

        logger.log_and_write(test_dict)
        logger.log_and_write("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
        logger.log_and_write("------------------------------------")
        dataset.reset_label_time()

    max_test_score = test_ndcgs[best_test_idx]
    logger.log_and_write("------------------------------------")
    logger.log_and_write("------------------------------------")
    logger.log_and_write("best val score: {}".format(max_val_score))
    logger.log_and_write("best validation epoch   : {}".format(best_test_idx + 1))
    logger.log_and_write("best test score: {}".format( max_test_score))

    wandb_summary = wandb_run.summary if wandb_run is not None else wandb.run.summary
    wandb_summary["best_val"] = max_val_score
    wandb_summary["best_test"] = max_test_score
    _sync_wandb_run()


if __name__ == '__main__':
    parser = utils.get_parser()
    args = parser.parse_args()
    main(args)
