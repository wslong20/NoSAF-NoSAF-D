import argparse
import random
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import homophily
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils.logger import Logger
from models import NoSAF, DeepNoSAF


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
    parser = argparse.ArgumentParser(description='ADGN')

    # training settings
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)

    # dataset settings
    parser.add_argument('--dataset_dir', type=str, default="/home/ssd7T/remotewsl/GraphDataset/ogbn_pyg")

    # input map and output map settings
    parser.add_argument('--input_map_numlayer', type=int, default=2)
    parser.add_argument('--input_map_norm', type=str, default="batchnorm", choices=["batchnorm", "none"])
    parser.add_argument('--output_map_numlayer', type=int, default=2)
    parser.add_argument('--output_map_norm', type=str, default="none", choices=["batchnorm", "none"])

    # backbone settings
    parser.add_argument('--use_acc_res', type=bool, default=True)
    parser.add_argument('--backbone_numlayer', type=int, default=32)
    parser.add_argument('--gnn_in_features', type=int, default=128)
    parser.add_argument('--backbone_norm', type=str, default="batchnorm", choices=["batchnorm", "none"])
    parser.add_argument('--learner_numlayer', type=int, default=2)
    parser.add_argument('--learner_hid_features', type=int, default=4)
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

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.dataset_dir, transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    x = data.x

    x = x.to(device)
    adj_t = data.adj_t.to(device)
    y_true = data.y.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    if not args.use_acc_res:
        model = NoSAF(in_features=x.size(-1),
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
                )
    else:
        model = DeepNoSAF(in_features=x.size(-1),
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
                )
    print(model)
    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else "cpu"
    # model = model.to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    idxs = torch.cat([train_idx])
    for run in range(args.runs):
        torch.cuda.empty_cache()
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_valid = 0
        best_out = None

        import time
        begin = time.time()
        for epoch in range(1, args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, adj_t)[idxs]
            loss = F.cross_entropy(out, y_true.squeeze(1)[idxs])
            result = test(model, x, y_true, adj_t, split_idx, evaluator)
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
