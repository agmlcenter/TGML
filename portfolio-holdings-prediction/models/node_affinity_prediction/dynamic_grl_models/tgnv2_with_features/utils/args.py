import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='random seed to use')
    parser.add_argument('--dataset', type=str, default='tgbn-13f', help="The dataset to train on.")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=200, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate.")
    parser.add_argument("--global_hidden_dims", type=int, default=512, help="Hidden dims for intermediate embeddings.")
    parser.add_argument("--num_last_neighbours", type=int, default=10, help="No. of last neighbours in GNN component.")
    parser.add_argument("--use_tgnv2", default=True, action=argparse.BooleanOptionalAction, help="Whether to use TGNv2 or not.")
    parser.add_argument("--learning_scheduler", type=str, default='constant')
    parser.add_argument("--cosine_annealing_ratio", type=float, default=0.2)
    parser.add_argument("--step_lr_gamma", type=float, default=0.5)
    parser.add_argument("--step_lr_step_size", type=int, default=250)
    return parser
