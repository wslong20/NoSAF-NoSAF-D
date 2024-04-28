import __init__
import torch
import torch.optim as optim
import statistics
from dataset import OGBNDataset
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from utils.data_util import intersection, process_indexes
import argparse
from model import NoSAF, DeepNoSAF
from logger import Logger


def train(data, dataset, model, optimizer, criterion, device):
    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)

        target = train_y[inter_idx].to(device)

        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator, device):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        test_target_idx = []

        train_predict = []
        valid_predict = []

        train_target_idx = []
        valid_target_idx = []

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().to(device)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

            pred = model(x, sg_nodes_idx, sg_edges[idx].to(device), sg_edges_attr).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(sg_nodes[idx], test_idx)
            test_target_idx += inter_te_idx

            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(train_pre_ordered_list), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(valid_pre_ordered_list), dim=0)
    test_pre_final = torch.mean(torch.Tensor(test_pre_ordered_list), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')

    # training settings
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_evals', type=int, default=1,
                        help='The number of evaluation times')


    # dataset settings
    parser.add_argument('--dataset_dir', type=str, default="/home/ssd7T/remotewsl/GraphDataset/ogbn_pyg")
    parser.add_argument('--dataset', type=str, default="ogbn-proteins")
    parser.add_argument('--num_tasks', type=int, default=112)
    parser.add_argument('--cluster_number', type=int, default=10,
                        help='the number of sub-graphs for training')
    parser.add_argument('--valid_cluster_number', type=int, default=6,
                        help='the number of sub-graphs for evaluation')
    parser.add_argument('--aggr', type=str, default='add',
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--nf_path', type=str, default='init_node_features_add.pt',
                        help='the file path of extracted node features saved.')

    # input map and output map settings

    parser.add_argument('--input_map_numlayer', type=int, default=1)
    parser.add_argument('--input_map_norm', type=str, default="none", choices=["batchnorm", "none"])
    parser.add_argument('--output_map_numlayer', type=int, default=1)
    parser.add_argument('--output_map_norm', type=str, default="none", choices=["batchnorm", "none"])

    # backbone settings
    parser.add_argument('--use_one_hot_encoding', action="store_true", default=True)
    parser.add_argument('--cpm', type=bool, default=True, help="compensatory mechanism")
    parser.add_argument('--backbone_numlayer', type=int, default=16)
    parser.add_argument('--gnn_in_features', type=int, default=256)
    parser.add_argument('--backbone_norm', type=str, default="batchnorm", choices=["batchnorm", "none"])
    parser.add_argument('--learner_numlayer', type=int, default=2)
    parser.add_argument('--learner_hid_features', type=int, default=64)
    parser.add_argument('--mlp_layers', type=int, default=2,
                        help='the number of layers of mlp in conv')
    # parser.add_argument('--conv', type=str, default='gen',
    #                     help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='softmax',
                        help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, softmax_sum, power, power_sum]')
    parser.add_argument('--norm', type=str, default='layer',
                        help='the type of normalization layer')
    # learnable parameters
    parser.add_argument('--t', type=float, default=1.0,
                        help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0,
                        help='the power of PowerMean')
    parser.add_argument('--y', type=float, default=0.0,
                        help='the power of degrees')
    parser.add_argument('--learn_t', action='store_true', default=True)
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--learn_y', action='store_true')
    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')
    # encode edge in conv
    parser.add_argument('--conv_encode_edge', action='store_true', default=True)
    # if use one-hot-encoding node feature

    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print("dataset_generator")
    dataset = OGBNDataset(dataset_name=args.dataset, root="/home/ssd7T/remotewsl/GraphDataset/ogbn_pyg")
    print("dataset_generator finished")
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)
    print("extract_node_features over")

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path
    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []

    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                               cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts,
                                                 cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    if not args.cpm:
        model = NoSAF(args)
    else:
        model = DeepNoSAF(args)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        torch.cuda.empty_cache()
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            # do random partition every epoch
            train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
                                                         cluster_number=args.cluster_number)
            data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)
            epoch_loss = train(data, dataset, model, optimizer, criterion, device)
            print('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))

            result = multi_evaluate(valid_data_list, dataset, model, evaluator, device)
            # results = result["train"], result["valid"], result["test"]

            if epoch % 5 == 0:
                print('%s' % result)

            train_result = result['train']['rocauc']
            valid_result = result['valid']['rocauc']
            test_result = result['test']['rocauc']
            result = train_result, valid_result, test_result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {epoch_loss:.4f}, '
                  f'Train: {100 * train_result:.2f}%, '
                  f'Valid: {100 * valid_result:.2f}% '
                  f'Test: {100 * test_result:.2f}%')
            logger.add_result(run, result)
            torch.cuda.empty_cache()
        end_time = time.time()
        total_time = end_time - start_time
        print('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
        logger.print_statistics(run)
    logger.print_statistics()

    # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(device)
    #
    # dataset = OGBNDataset(dataset_name="ogbn-proteins", root="/home/ssd7T/remotewsl/GraphDataset/ogbn_pyg")
    # # extract initial node features
    # nf_path = dataset.extract_node_features(args.aggr)
    #
    # args.num_tasks = dataset.num_tasks
    # args.nf_path = nf_path
    #
    # evaluator = Evaluator("ogbn-proteins")
    # criterion = torch.nn.BCEWithLogitsLoss()
    #
    # model = NoSAF(args)
    # print(model)
    # for run in range(args.runs):
    #     torch.cuda.empty_cache()
    #     print(sum(p.numel() for p in model.parameters()))
    #     model.reset_parameters()
    #     model = model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     start_time = time.time()
    #     for epoch in range(1, args.epochs + 1):
    #         # do random partition every epoch
    #         train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes,
    #                                                      cluster_number=args.cluster_number)
    #         data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)
    #
    #         epoch_loss = train(data, dataset, model, optimizer, criterion, device)
    #         result = test(model, data, split_idx, evaluator)
    #
    #         if epoch % 5 == 0:
    #             logging.info('%s' % result)
    #
    #         train_result = result['train']['rocauc']
    #         valid_result = result['valid']['rocauc']
    #         test_result = result['test']['rocauc']
    #
    #         if valid_result > results['highest_valid']:
    #             results['highest_valid'] = valid_result
    #             results['final_train'] = train_result
    #             results['final_test'] = test_result
    #
    #             save_ckpt(model, optimizer, round(epoch_loss, 4),
    #                       epoch,
    #                       args.model_save_path, sub_dir,
    #                       name_post='valid_best')
    #
    #         if train_result > results['highest_train']:
    #             results['highest_train'] = train_result
    #
    # logging.info("%s" % results)
    #
    # end_time = time.time()
    # total_time = end_time - start_time
    # logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
