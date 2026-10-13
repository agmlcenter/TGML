"""
DyRep
    This has been implemented with intuitions from the following sources:
    - https://github.com/twitter-research/tgn
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

    Spec.:
        - Memory Updater: RNN
        - Embedding Module: ID
        - Message Function: ATTN
"""
import os
dir_name = os.path.dirname(os.path.realpath(__file__))
four_folders_back = os.path.normpath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(four_folders_back)

import timeit
from tqdm import tqdm
import torch
from torch_geometric.loader import TemporalDataLoader
import wandb
import subprocess
import atexit

# internal imports
from tgb.utils.utils import get_args, set_random_seed
from tgb.nodeproppred.evaluate import Evaluator
from modules.decoder import NodePredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import DyRepMemory
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset


def process_edges(src, dst, t, msg):
    if src.nelement() > 0:
        model['memory'].update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


# ==========
# ========== Define helper function...
# ==========

def train(epoch):
    model['memory'].train()
    model['gnn'].train()
    model['node_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
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

            z, last_update = model['memory'](n_id_neighbors)
            z = model['gnn'](
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            # loss and metric computation
            pred = model['node_pred'](z)

            loss = criterion(pred, labels.to(device))
            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[metric]
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
        model['memory'].detach()

    metric_dict = {
        "ce": total_loss / num_label_ts,
    }
    metric_dict[metric] = total_score / num_label_ts
    return metric_dict


@torch.no_grad()
def test(loader):
    model['memory'].eval()
    model['gnn'].eval()
    model['node_pred'].eval()

    total_score = 0
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

            z, last_update = model['memory'](n_id_neighbors)
            z = model['gnn'](
                z,
                last_update,
                mem_edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            z = z[assoc[n_id]]

            pred = model['node_pred'](z)
            loss = criterion(pred, labels.to(device))
            total_loss += float(loss)

            np_pred = pred.cpu().detach().numpy()
            np_true = labels.cpu().detach().numpy()
            result_dict = evaluator.eval({
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [metric],
            })
            total_score += result_dict[metric]
            num_label_ts += 1

        process_edges(src, dst, t, msg)

    metric_dict = {}
    if num_label_ts > 0:
        metric_dict["ce"] = total_loss / num_label_ts
        metric_dict[metric] = total_score / num_label_ts
    else:
        metric_dict["ce"] = 0.0
        metric_dict[metric] = 0.0
    return metric_dict

# ==========
# ==========
# ==========

# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()

DATA = "tgbn-13f"
args.data = DATA
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
print("INFO: Arguments:", args)

run_name = f"dyrep_dataset_{DATA}_lr_{LR}_bs_{BATCH_SIZE}_epochs_{NUM_EPOCH}_seed_{SEED}"
wandb_config = {
    "dataset": DATA,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCH,
    "seed": SEED,
    "mem_dim": MEM_DIM,
    "time_dim": TIME_DIM,
    "emb_dim": EMB_DIM,
}
wandb_kwargs = {
    "project": "dyrep_without_feature",
    "entity": "emad-izadifar-amirkabir-university-of-technology",
    "name": run_name,
    "config": wandb_config,
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

# setting random seed
torch.manual_seed(SEED)
set_random_seed(SEED)

MODEL_NAME = 'DyRep'
USE_SRC_EMB_IN_MSG = False
USE_DST_EMB_IN_MSG = True
evaluator = Evaluator(name=DATA)
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
dataset = PyGNodePropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]
num_classes = dataset.num_classes


train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighborhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = DyRepMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
    memory_updater_type='rnn',
    use_src_emb_in_msg=USE_SRC_EMB_IN_MSG,
    use_dst_emb_in_msg=USE_DST_EMB_IN_MSG
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

node_pred = NodePredictor(in_dim=EMB_DIM, out_dim=num_classes).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'node_pred': node_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['node_pred'].parameters()),
    lr=LR,
)

criterion = torch.nn.CrossEntropyLoss()
# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

train_curve = []
val_curve = []
test_curve = []
max_val_score = 0  #find the best test score based on validation score
best_test_idx = 0
for epoch in range(1, NUM_EPOCH + 1):
    start_time = timeit.default_timer()
    train_dict = train(epoch)
    print("------------------------------------")
    print(f"training Epoch: {epoch:02d}")
    print(train_dict)
    train_curve.append(train_dict[metric])
    wandb.log({
        "train/loss": train_dict["ce"],
        "train/train_loss": train_dict["ce"],
        f"train/{metric}": train_dict[metric],
        "train/epoch": epoch,
    })
    print("Training takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    
    start_time = timeit.default_timer()
    val_dict = test(val_loader)
    print(val_dict)
    val_curve.append(val_dict[metric])
    wandb.log({
        f"val/{metric}": val_dict[metric],
        "val/val_loss": val_dict["ce"],
        "val/epoch": epoch,
    })
    if (val_dict[metric] > max_val_score):
        max_val_score = val_dict[metric]
        best_test_idx = epoch - 1
    print("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    test_dict = test(test_loader)
    print(test_dict)
    test_curve.append(test_dict[metric])
    wandb.log({
        f"test/{metric}": test_dict[metric],
        "test/test_loss": test_dict["ce"],
        "test/epoch": epoch,
    })
    print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    print("------------------------------------")
    dataset.reset_label_time()

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

