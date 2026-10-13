import os
dir_name = os.path.dirname(os.path.realpath(__file__))
four_folders_back = os.path.normpath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(four_folders_back)

import timeit
import argparse
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import wandb
import math
import subprocess
import atexit

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed
from tgb.utils.stats import plot_curve

parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use.')
parser.add_argument('--dataset', type=str, default='tgbn-13f', help='Dataset to train on.')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=200, help='Mini-batch size.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--global_hidden_dims', type=int, default=512, help='Hidden dimension size.')
parser.add_argument('--num_last_neighbours', type=int, default=10, help='Number of last neighbours to sample.')
parser.add_argument('--learning_scheduler', type=str, default='constant', choices=['constant', 'cosine_annealing', 'step_lr'], help='Type of LR scheduler.')
parser.add_argument('--cosine_annealing_ratio', type=float, default=0.2, help='Cosine annealing ratio of total epochs.')
parser.add_argument('--step_lr_gamma', type=float, default=0.5, help='Gamma for StepLR scheduler.')
parser.add_argument('--step_lr_step_size', type=int, default=250, help='Step size for StepLR scheduler.')
args = parser.parse_args()
# setting random seed
seed = int(args.seed)  # 1,2,3,4,5
torch.manual_seed(seed)
set_random_seed(seed)

lr = args.lr
epochs = args.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = args.dataset
dataset = PyGNodePropPredDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

eval_metric = dataset.eval_metric
num_classes = dataset.num_classes
data = dataset.get_TemporalData()
data = data.to(device)

evaluator = Evaluator(name=name)


train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

batch_size = args.batch_size

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=args.num_last_neighbours, device=device)

memory_dim = time_dim = embedding_dim = args.global_hidden_dims

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = (
    GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    )
    .to(device)
    .float()
)

node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters()) | set(node_pred.parameters()),
    lr=lr,
)

if args.learning_scheduler == 'constant':
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scheduler_desc = ''
elif args.learning_scheduler == 'cosine_annealing':
    ratio = args.cosine_annealing_ratio
    t_max = int(math.ceil(args.epochs / ratio))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler_desc = f'_ratio_{ratio}'
elif args.learning_scheduler == 'step_lr':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_lr_step_size,
        gamma=args.step_lr_gamma,
    )
    scheduler_desc = f'_gamma_{args.step_lr_gamma}_step_size_{args.step_lr_step_size}'
else:
    raise ValueError(f"Unsupported learning scheduler: {args.learning_scheduler}")

criterion = torch.nn.CrossEntropyLoss()
# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def plot_curve(scores, out_name):
    import matplotlib
    matplotlib.use("Agg")
    plt.plot(scores, color="#e34a33")
    plt.ylabel("score")
    plt.savefig(out_name + ".pdf")
    plt.close()


def process_edges(src, dst, t, msg):
    if src.nelement() > 0:
        # msg = msg.to(torch.float32)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def train(epoch, lr_scheduler):
    memory.train()
    gnn.train()
    node_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0

    for batch in train_loader:
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
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)

            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
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

            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch,
            })

        # Update memory and neighbor loader with ground-truth state.
        process_edges(src, dst, t, msg)
        memory.detach()

    metric_dict = {
        "ce": total_loss / num_label_ts,
        eval_metric: total_score / num_label_ts,
    }
    if lr_scheduler is not None:
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
    else:
        current_lr = optimizer.param_groups[0]["lr"]
    metric_dict["lr"] = current_lr
    return metric_dict


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    node_pred.eval()

    total_score = 0.0
    total_loss = 0.0
    label_t = dataset.get_label_time()
    num_label_ts = 0

    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = label_tuple
            label_t = dataset.get_label_time()
            label_srcs = label_srcs.to(device)

            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask],
                dst[previous_day_mask],
                t[previous_day_mask],
                msg[previous_day_mask],
            )
            src, dst, t, msg = (
                src[~previous_day_mask],
                dst[~previous_day_mask],
                t[~previous_day_mask],
                msg[~previous_day_mask],
            )

            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)
            z = gnn(
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

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
            total_score += result_dict[eval_metric]
            num_label_ts += 1

        process_edges(src, dst, t, msg)

    if num_label_ts == 0:
        return {"ce": 0.0, eval_metric: 0.0}

    return {
        "ce": total_loss / num_label_ts,
        eval_metric: total_score / num_label_ts,
    }


run_name = (
    f"scheduler_{args.learning_scheduler}{scheduler_desc}_tgn_dataset_{name}_bs_{batch_size}"
    f"_lr_{lr}_epochs_{epochs}_last_neighbour_{args.num_last_neighbours}"
    f"_global_dims_{args.global_hidden_dims}_seed_{seed}"
)
wandb_kwargs = {
    "project": "tgn_without_feature",
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

train_curve = []
val_curve = []
test_curve = []
max_val_score = 0  #find the best test score based on validation score
best_test_idx = 0
for epoch in range(1, epochs + 1):
    start_time = timeit.default_timer()
    train_dict = train(epoch=epoch, lr_scheduler=lr_scheduler)
    print("------------------------------------")
    print(f"training Epoch: {epoch:02d}")
    print(train_dict)
    train_curve.append(train_dict[eval_metric])
    wandb.log({
        "train/loss": train_dict["ce"],         # renamed from train/ce
        "train/train_loss": train_dict["ce"],   # epoch-level average loss
        f"train/{eval_metric}": train_dict[eval_metric],
        "train/lr": train_dict["lr"],
        "train/epoch": epoch,                   # put epoch under train
    })
    print("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    
    start_time = timeit.default_timer()
    val_dict = test(val_loader)
    print(val_dict)
    val_curve.append(val_dict[eval_metric])
    wandb.log({
        f"val/{eval_metric}": val_dict[eval_metric],
        "val/val_loss": val_dict["ce"],
        "val/epoch": epoch,
    })
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
    print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    print("------------------------------------")
    dataset.reset_label_time()


# code for plotting
plot_curve(train_curve, "train_curve")
plot_curve(val_curve, "val_curve")
plot_curve(test_curve, "test_curve")

max_test_score = test_curve[best_test_idx]
print("------------------------------------")
print("------------------------------------")
print ("best val score: ", max_val_score)
print ("best validation epoch   : ", best_test_idx + 1)
print ("best test score: ", max_test_score)

wandb_summary = wandb_run.summary if wandb_run is not None else wandb.run.summary
wandb_summary["best_val"] = max_val_score
wandb_summary["best_test"] = max_test_score
_sync_wandb_run()
