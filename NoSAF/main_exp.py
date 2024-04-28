import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.transforms import RandomNodeSplit
from ogb.nodeproppred import Evaluator

from utils.logger import Logger
from models import NoSAF, DeepNoSAF
from utils.utils import get_dataset


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y, adj, split_idx, evaluator):
    model.eval()
    y = y.unsqueeze(-1)

    out = model(x, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='Experiment on homophilic and heterophilic graph')

    # training settings
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)

    # dataset settings
    # parser.add_argument('--dataset_dir', type=str, default="/home/ssd7T/remotewsl/GraphDataset/heterophilic_graph_pyg")
    parser.add_argument('--dataset_dir', type=str, default="/home/ssd7T/remotewsl/GraphDataset/homophilic_graph_pyg")
    parser.add_argument('--dataset', type=str, default="Physics",
                        choices=["Cora", "CiteSeer", "PubMed", "CoraFull", "Computers", "Photo", "CS", "Physics", "WikiCS",
                                 "Cornell", "Texas", "Wisconsin", "Chameleon", "Squirrel", "Actor"])

    # input map and output map settings
    parser.add_argument('--input_map_numlayer', type=int, default=2)
    parser.add_argument('--input_map_norm', type=str, default="batchnorm", choices=["batchnorm", "none"])
    parser.add_argument('--output_map_numlayer', type=int, default=1)
    parser.add_argument('--output_map_norm', type=str, default="none", choices=["batchnorm", "none"])

    # backbone settings
    parser.add_argument('--Deep', action="store_true", help="If True, NoSAF-D")
    parser.add_argument('--backbone_numlayer', type=int, default=64)
    parser.add_argument('--gnn_in_features', type=int, default=128)
    parser.add_argument('--backbone_norm', type=str, default="batchnorm", choices=["batchnorm", "none"])
    parser.add_argument('--learner_numlayer', type=int, default=2)
    parser.add_argument('--learner_hid_features', type=int, default=32)
    parser.add_argument('--negative_slope', type=float, default=0.2)

    args = parser.parse_args()
    print(args)

    if args.seed != -1:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset.lower() == "corafull":
        args.dataset_dir = os.path.join(args.dataset_dir, "corafull")

    dataset = get_dataset(root=args.dataset_dir, name=args.dataset)

    data = dataset[0]
    # To undirected graph.
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    y_true = data.y.to(device)
    evaluator = Evaluator("ogbn-arxiv")

    if not args.Deep:
        model = NoSAF(in_features=data.num_features,
                       gnn_in_features=args.gnn_in_features,
                       out_features=dataset.num_classes,
                       input_map_numlayer=args.input_map_numlayer,
                       backbone_numlayer=args.backbone_numlayer,
                       output_map_numlayer=args.output_map_numlayer,
                       input_map_norm=args.input_map_norm,
                       backbone_norm=args.backbone_norm,
                       output_map_norm=args.output_map_norm,
                       learner_hid_features=args.learner_hid_features,
                       learner_numlayer=args.learner_numlayer,
                       negative_slope=args.negative_slope,
                       dropout=args.dropout,
                ).to(device)
    else:
        model = DeepNoSAF(in_features=data.num_features,
                             gnn_in_features=args.gnn_in_features,
                             out_features=dataset.num_classes,
                             input_map_numlayer=args.input_map_numlayer,
                             backbone_numlayer=args.backbone_numlayer,
                             output_map_numlayer=args.output_map_numlayer,
                             input_map_norm=args.input_map_norm,
                             backbone_norm=args.backbone_norm,
                             output_map_norm=args.output_map_norm,
                             learner_hid_features=args.learner_hid_features,
                             learner_numlayer=args.learner_numlayer,
                             negative_slope=args.negative_slope,
                             dropout=args.dropout,
                ).to(device)
    print(model)

    logger = Logger(args.runs, args)
    for run in range(args.runs):
        # random split the training set, verification set, and test set by 6/2/2
        split_idx = {}
        split_gen = RandomNodeSplit(split='train_rest', num_val=0.20, num_test=0.20)
        data = split_gen(data)
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        valid_idx = data.val_mask.nonzero(as_tuple=True)[0]
        test_idx = data.test_mask.nonzero(as_tuple=True)[0]
        split_idx["train"] = train_idx.to(device)
        split_idx["valid"] = valid_idx.to(device)
        split_idx["test"] = test_idx.to(device)
        # print(f"train set num: {len(train_idx)}, valid set num: {len(valid_idx)}, test set num: {len(test_idx)}")
        # torch.cuda.empty_cache()
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)[train_idx]
            loss = F.cross_entropy(out, y_true.squeeze(-1)[train_idx])
            result = test(model, x, y_true, edge_index, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result

            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            logger.add_result(run, result)
            loss.backward()
            optimizer.step()
        logger.print_statistics(run)

    logger.print_statistics()


if __name__ == "__main__":
    main()
