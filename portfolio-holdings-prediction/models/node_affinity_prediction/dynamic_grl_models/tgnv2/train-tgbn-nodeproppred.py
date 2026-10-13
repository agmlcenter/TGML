import timeit
import torch
import wandb
import os
import math
import utils
import subprocess
import atexit

from tqdm import tqdm
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        use_gnn=True):
    eval_metric = dataset.eval_metric

    memory.train()
    gnn.train()
    node_pred.train()

    criterion = torch.nn.CrossEntropyLoss()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    total_score = 0

    train_loader_length = len(train_loader)
    count = 0
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
            2. extract memory from the sampled neighbors and the nodes
            3. run gnn with the extracted memory embeddings and the corresponding time and message
            """
            n_id = label_srcs
            n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)

            z, last_update = memory(n_id_neighbors)
            if use_gnn:
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
        epoch: int,
        use_gnn=True):

    eval_metric = dataset.eval_metric

    memory.eval()
    gnn.eval()
    node_pred.eval()
    total_score = 0
    total_loss = 0.0
    label_t = dataset.get_label_time()  # check when does the first label start
    num_label_ts = 0
    criterion = torch.nn.CrossEntropyLoss()

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
            f"{split}/epoch": epoch,
        }
    else:
        metric_dict = {
            f"{split}/{eval_metric}": total_score / num_label_ts,
            f"{split}/{split}_loss": total_loss / num_label_ts,
            f"{split}/epoch": epoch,
        }
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
        "project": "tgnv2_without_feature",
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
            use_gnn=use_gnn
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
