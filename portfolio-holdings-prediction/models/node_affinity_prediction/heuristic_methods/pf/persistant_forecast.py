"""
implement persistant forecast as baseline for the node prop pred task
simply predict last seen label for the node
"""

import os
import atexit
import subprocess
dir_name = os.path.dirname(os.path.realpath(__file__))
three_folders_back = os.path.normpath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(three_folders_back)
import timeit
import numpy as np
from torch_geometric.loader import TemporalDataLoader

# local imports
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from modules.heuristics import PersistantForecaster
from tgb.nodeproppred.evaluate import Evaluator
import wandb

device = 'cpu'

name = 'tgbn-13f'
dataset = PyGNodePropPredDataset(name=name, root='datasets')
num_classes = dataset.num_classes
data = dataset.get_TemporalData()
data = data.to(device)

eval_metric = dataset.eval_metric
forecaster = PersistantForecaster(num_classes)
evaluator = Evaluator(name=name)

wandb_run = None
wandb_synced = False
wandb_desired_mode = os.getenv("WANDB_MODE", "online")
actual_wandb_mode = wandb_desired_mode


def _init_wandb():
    global wandb_run, actual_wandb_mode
    try:
        wandb_run = wandb.init(
            mode=wandb_desired_mode,
            project="pf",
            entity="emad-izadifar-amirkabir-university-of-technology",
            name=f"pf_{name}",
            config={
                "dataset": name,
                "batch_size": 200,
            },
        )
        actual_wandb_mode = getattr(getattr(wandb_run, "settings", None), "mode", wandb_desired_mode)
    except Exception as exc:
        if wandb_desired_mode.lower() != "offline":
            print(f"[wandb] Init failed in mode '{wandb_desired_mode}'. Falling back to offline. Error: {exc}")
            wandb_run = wandb.init(
                mode="offline",
                reinit=True,
                project="pf",
                entity="emad-izadifar-amirkabir-university-of-technology",
                name=f"pf_{name}",
                config={
                    "dataset": name,
                    "batch_size": 200,
                },
            )
            actual_wandb_mode = "offline"
        else:
            raise


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


_init_wandb()
atexit.register(_sync_wandb_run)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15
)

batch_size = 200

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)


def test_n_upate(loader, split_name: str):
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0
    last_epoch_marker = None

    for batch in loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        query_t = batch.t[-1]
        if query_t >= label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_ts = label_ts.numpy()
            label_srcs = label_srcs.numpy()
            labels = labels.numpy()
            label_t = dataset.get_label_time()
            if label_ts.size:
                last_epoch_marker = label_ts[-1]

            preds = []

            for i in range(0, label_srcs.shape[0]):
                node_id = label_srcs[i]
                pred_vec = forecaster.query_dict(node_id)
                preds.append(pred_vec)
                forecaster.update_dict(node_id, labels[i])

            np_pred = np.stack(preds, axis=0)
            np_true = labels

            input_dict = {
                'y_true': np_true,
                'y_pred': np_pred,
                'eval_metric': [eval_metric],
            }
            result_dict = evaluator.eval(input_dict)
            score = result_dict[eval_metric]
            total_score += score
            num_label_ts += 1
            epoch_val = int(last_epoch_marker) if last_epoch_marker is not None else num_label_ts
            wandb.log(
                {
                    f"{split_name}/{eval_metric}": score,
                    f"{split_name}/epoch": epoch_val,
                }
            )

    metric_dict = {}
    metric_dict[eval_metric] = total_score / num_label_ts if num_label_ts > 0 else float("nan")
    epoch_marker = int(last_epoch_marker) if last_epoch_marker is not None else num_label_ts
    return metric_dict, epoch_marker


"""
train, val and test for one epoch only
"""

start_time = timeit.default_timer()
metric_dict, train_epoch = test_n_upate(train_loader, "train")
print(metric_dict)
wandb.log({
    f"train/{eval_metric}": metric_dict[eval_metric],
    "train/epoch": int(train_epoch),
})
print(
    'Persistant forecast on Training takes--- %s seconds ---'
    % (timeit.default_timer() - start_time)
)

start_time = timeit.default_timer()
val_dict, val_epoch = test_n_upate(val_loader, "val")
print(val_dict)
wandb.log({
    f"val/{eval_metric}": val_dict[eval_metric],
    "val/epoch": int(val_epoch),
})
print(
    'Persistant forecast on validation takes--- %s seconds ---'
    % (timeit.default_timer() - start_time)
)


start_time = timeit.default_timer()
test_dict, test_epoch = test_n_upate(test_loader, "test")
print(test_dict)
wandb.log({
    f"test/{eval_metric}": test_dict[eval_metric],
    "test/epoch": int(test_epoch),
})
print(
    'Persistant forecast on Test takes--- %s seconds ---' % (timeit.default_timer() - start_time)
)
dataset.reset_label_time()
_sync_wandb_run()