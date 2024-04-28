from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn import Linear
from gcn_lib.sparse.torch_vertex import GENConv
from models import GCNBlock, NodeWeightLearner, MappingNN
from gcn_lib.sparse.torch_nn import norm_layer


class NoSAF(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.input_map_numlayer < 1 or args.backbone_numlayer < 1 or args.output_map_numlayer < 1:
            raise ValueError("numlayer cannot be less than 1")
        self.input_map_numlayer = args.input_map_numlayer  # TODO
        self.backbone_numlayer = args.backbone_numlayer
        self.output_map_numlayer = args.output_map_numlayer  #

        self.gnn_in_features = args.gnn_in_features
        self.learner_hid_features = args.learner_hid_features
        self.learner_numlayer = args.learner_numlayer

        self.dropout = args.dropout
        num_tasks = args.num_tasks

        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm  # False
        learn_msg_scale = args.learn_msg_scale  # False

        conv_encode_edge = args.conv_encode_edge  # True
        norm = args.norm  # layer
        mlp_layers = args.mlp_layers  # 2
        node_features_file_path = args.nf_path
        if self.backbone_numlayer > 4:
            self.checkpoint_grad = True
            self.ckp_k = 4
        self.use_one_hot_encoding = args.use_one_hot_encoding  # True

        print('The number of layers {}'.format(self.backbone_numlayer),
              'Aggregation method {}'.format(aggr))

        self.gcns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.learners = nn.ModuleList()
        self.learners.append(
            NodeWeightLearner(in_features=self.gnn_in_features,
                              hid_features=self.learner_hid_features,
                              numlayer=self.learner_numlayer,
                              negative_slope=0.2)
        )  # input learner
        for layer in range(self.backbone_numlayer):
            gcn = GENConv(self.gnn_in_features, self.gnn_in_features,
                          aggr=aggr,
                          t=t, learn_t=self.learn_t,
                          p=p, learn_p=self.learn_p,
                          y=y, learn_y=self.learn_y,
                          msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                          encode_edge=conv_encode_edge, edge_feat_dim=self.gnn_in_features,
                          norm=norm, mlp_layers=mlp_layers)
            self.gcns.append(gcn)
            self.learners.append(
                NodeWeightLearner(in_features=self.gnn_in_features,
                                  hid_features=self.learner_hid_features,
                                  numlayer=self.learner_numlayer,
                                  negative_slope=0.2)
            )
            self.layer_norms.append(norm_layer(norm, self.gnn_in_features))

        self.node_features = torch.load(node_features_file_path).to(args.device)

        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = Linear(8, 8)
            self.node_features_encoder = Linear(8 * 2, self.gnn_in_features)
            # self.node_features_encoder = MappingNN(
            #     in_features=8 * 2,
            #     hid_features=self.gnn_in_features * 2,
            #     out_features=self.gnn_in_features,
            #     numlayer=self.input_map_numlayer,
            #     activation="relu",
            #     norm="none",
            #     dropout=self.dropout
            # )
        else:
            # self.node_features_encoder = MappingNN(
            #     in_features=8,
            #     hid_features=self.gnn_in_features * 2,
            #     out_features=self.gnn_in_features,
            #     numlayer=self.input_map_numlayer,
            #     activation="relu",
            #     norm="batchnorm",
            #     dropout=self.dropout
            # )
            self.node_features_encoder = Linear(8, self.gnn_in_features)


        self.edge_encoder = Linear(8, self.gnn_in_features)
        self.node_pred_linear = Linear(self.gnn_in_features, num_tasks)
        # self.node_pred_linear = MappingNN(
        #     in_features=self.gnn_in_features,
        #     hid_features=self.gnn_in_features * 2,
        #     out_features=num_tasks,
        #     numlayer=self.output_map_numlayer,
        #     activation="relu",
        #     norm="none",
        #     dropout=self.dropout
        # )
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.learners)):
            self.learners[i].reset_parameters()
        for i in range(len(self.gcns)):
            self.gcns[i].reset_parameters()
        if self.use_one_hot_encoding:
            self.node_one_hot_encoder.reset_parameters()
        self.node_features_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_pred_linear.reset_parameters()

    def forward(self, x, node_index, edge_index, edge_attr):
        node_features_1st = self.node_features[node_index]
        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            # concatenate
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st

        if self.checkpoint_grad:
            h = checkpoint(self.node_features_encoder, node_features)
            nw = checkpoint(self.learners[0], h)
            edge_emb = checkpoint(self.edge_encoder, edge_attr)
        else:
            h = self.node_features_encoder(node_features)
            nw = self.learners[0](h)  # node weight
            edge_emb = self.edge_encoder(edge_attr)
        h = h * nw
        codebank = h * nw
        # h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb)))
        # nw = self.learners[1](h, codebank)
        # h = h * nw
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # codebank = codebank + h
        # if self.checkpoint_grad:
        for layer in range(0, self.backbone_numlayer):
            h1 = checkpoint(self.gcns[layer], h, edge_index, edge_emb)
            h2 = self.layer_norms[layer](h1)
            h = F.relu(h2)
            nw = checkpoint(self.learners[layer + 1], h, codebank)
            h_filtered = h * nw
            h = h_filtered
            codebank = codebank + h_filtered
            h = F.dropout(h, p=self.dropout, training=self.training)
            # out = checkpoint(self.node_pred_linear, codebank)
        # else:
        #     for layer in range(0, self.backbone_numlayer):
        #         h1 = self.gcns[layer](h, edge_index, edge_emb)
        #         h2 = self.layer_norms[layer](h1)
        #         h = F.relu(h2)
        #         nw = self.learners[layer + 1](h, codebank)
        #         h = h * nw
        #         codebank = codebank + h
        #         h = F.dropout(h, p=self.dropout, training=self.training)
            # out = self.node_pred_linear(codebank)
        return self.node_pred_linear(codebank)


class DeepNoSAF(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.input_map_numlayer < 1 or args.backbone_numlayer < 1 or args.output_map_numlayer < 1:
            raise ValueError("numlayer cannot be less than 1")
        self.input_map_numlayer = args.input_map_numlayer  # TODO
        self.backbone_numlayer = args.backbone_numlayer
        self.output_map_numlayer = args.output_map_numlayer  #

        self.gnn_in_features = args.gnn_in_features
        self.learner_hid_features = args.learner_hid_features
        self.learner_numlayer = args.learner_numlayer

        self.dropout = args.dropout
        self.checkpoint_grad = False
        num_tasks = args.num_tasks

        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm  # False
        learn_msg_scale = args.learn_msg_scale  # False

        conv_encode_edge = args.conv_encode_edge  # True
        norm = args.norm  # layer
        mlp_layers = args.mlp_layers  # 2
        node_features_file_path = args.nf_path

        self.use_one_hot_encoding = args.use_one_hot_encoding  # True

        print('The number of layers {}'.format(self.backbone_numlayer),
              'Aggregation method {}'.format(aggr))

        if aggr not in ['add', 'max', 'mean'] and self.backbone_numlayer > 8:
            self.checkpoint_grad = True
            # self.ckp_k = 9
        self.gcns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.learners = nn.ModuleList()
        self.learners.append(
            NodeWeightLearner(in_features=self.gnn_in_features,
                              hid_features=self.learner_hid_features,
                              numlayer=self.learner_numlayer,
                              negative_slope=0.2)
        )  # input learner
        for layer in range(self.backbone_numlayer):
            gcn = GENConv(self.gnn_in_features, self.gnn_in_features,
                          aggr=aggr,
                          t=t, learn_t=self.learn_t,
                          p=p, learn_p=self.learn_p,
                          y=y, learn_y=self.learn_y,
                          msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                          encode_edge=conv_encode_edge, edge_feat_dim=self.gnn_in_features,
                          norm=norm, mlp_layers=mlp_layers)
            self.gcns.append(gcn)
            self.learners.append(
                NodeWeightLearner(in_features=self.gnn_in_features,
                                  hid_features=self.learner_hid_features,
                                  numlayer=self.learner_numlayer,
                                  negative_slope=0.2)
            )
            self.layer_norms.append(norm_layer(norm, self.gnn_in_features))

        self.node_features = torch.load(node_features_file_path).to(args.device)

        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(8, 8)
            self.node_features_encoder = torch.nn.Linear(8 * 2, self.gnn_in_features)
        else:
            self.node_features_encoder = torch.nn.Linear(8, self.gnn_in_features)

        self.edge_encoder = torch.nn.Linear(8, self.gnn_in_features)
        self.node_pred_linear = torch.nn.Linear(self.gnn_in_features, num_tasks)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.learners)):
            self.learners[i].reset_parameters()
        for i in range(len(self.gcns)):
            self.gcns[i].reset_parameters()
        if self.use_one_hot_encoding:
            self.node_one_hot_encoder.reset_parameters()
        self.node_features_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_pred_linear.reset_parameters()

    def forward(self, x, node_index, edge_index, edge_attr):
        node_features_1st = self.node_features[node_index]
        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            # concatenate
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st

        h = checkpoint(self.node_features_encoder, node_features)
        # h = self.node_features_encoder(node_features)
        # nw = self.learners[0](h)  # node weight
        nw = checkpoint(self.learners[0], h)
        h = h * nw
        codebank = h * nw
        # edge_emb = self.edge_encoder(edge_attr)
        edge_emb = checkpoint(self.edge_encoder, edge_attr)

        # h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb)))
        # nw = self.learners[1](h, codebank)
        # h_filtered = h * nw
        # h = h_filtered + codebank * (1 - nw)
        # codebank = codebank + h_filtered
        h = F.dropout(h, p=self.dropout, training=self.training)
        for layer in range(0, self.backbone_numlayer):
            h1 = checkpoint(self.gcns[layer], h, edge_index, edge_emb)
            h2 = self.layer_norms[layer](h1)
            h = F.relu(h2)
            # nw = self.learners[layer+1](h, codebank)
            nw = checkpoint(self.learners[layer+1], h, codebank)
            h_filtered = h * nw
            h = h_filtered + codebank * (1 - nw)
            codebank = codebank + h_filtered
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.node_pred_linear(codebank)