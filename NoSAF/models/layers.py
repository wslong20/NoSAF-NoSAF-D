from typing import List, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import Linear
import torch
from torch_geometric.nn.aggr import Aggregation


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: str = None, norm: str = None, negative_slope: float = 0.01):
        """
        :param activation: str, "relu", "leaky_relu", "tanh", "none" or None
        :param norm: str, "batchnorm", "none" or None
        :param negative_slope: float, negative_slope of leaky_relu, if activation != "leaky_relu", this parameter is invalid
        """
        super().__init__()
        # self.lin = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.lin = Linear(in_channels=in_features, out_channels=out_features, bias=bias)
        self.activation = lambda x: x
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
            self.negative_slope = negative_slope
        elif activation == "none" or activation is None:
            pass
        else:
            raise NotImplementedError("Activation function can only be ('relu', 'tanh', 'none', None)")

        self.norm = lambda x: x
        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "none" or norm is None:
            pass
        else:
            raise NotImplementedError("normalization can only be ('batchnorm', 'none', None)")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if isinstance(self.norm, nn.BatchNorm1d):
            self.norm.reset_parameters()

    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        if self.activation == F.leaky_relu:
            x = self.activation(x, negative_slope=self.negative_slope)
        else:
            x = self.activation(x)
        return x


class GCNBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = None, norm: str = None, negative_slope: float = 0.01,
                 improved: bool=False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True, **kwargs):
        """
        improved, cached, add_self_loops, normalize, **kwargs are parameters of GCNConv
        :param activation: str, "relu", "leaky_relu", "tanh", "none" or None
        :param norm: str, "batchnorm", "none" or None
        :param negative_slope: float, negative_slope of leaky_relu, if activation != "leaky_relu", this parameter is invalid
        """
        super().__init__()
        self.gnn = GCNConv(in_features,
                           out_features,
                           bias=bias,
                           improved=improved,
                           cached=cached,
                           add_self_loops=add_self_loops,
                           normalize=normalize,
                           **kwargs)

        self.activation = lambda x: x
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
            self.negative_slope = negative_slope
        elif activation == "none" or activation is None:
            pass
        else:
            raise NotImplementedError("Activation function can only be ('relu', 'tanh', 'none', None)")

        self.norm = lambda x: x
        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "none" or norm is None:
            pass
        else:
            raise NotImplementedError("normalization can only be ('batchnorm', 'none', None)")

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if isinstance(self.norm, nn.BatchNorm1d):
            self.norm.reset_parameters()

    def forward(self, x, adj):
        x = self.gnn(x, adj)
        x = self.norm(x)
        if self.activation == F.leaky_relu:
            x = self.activation(x, negative_slope=self.negative_slope)
        else:
            x = self.activation(x)
        return x


class NodeWeightLearner(nn.Module):
    def __init__(self, in_features: int, hid_features: int, numlayer: int = 2, negative_slope: float = 0.2):
        """
        :param in_features: input dim of extractor
        :param hid_features: output dim of extractor
        :param num_layer: num layer of learner
        :param negative_slope: negative slope of LeakyRelu
        """
        super().__init__()
        if numlayer < 1:
            raise ValueError("numlayer of Node Weight Learner cannot be less than 1")
        self.numlayer = numlayer
        # self.extractor_1 = nn.Linear(in_features, hid_features, bias=False)
        # self.extractor_2 = nn.Linear(in_features, hid_features, bias=False)
        self.extractor_1 = Linear(in_features, hid_features, bias=False)
        self.extractor_2 = Linear(in_features, hid_features, bias=False)
        learner_in_features = hid_features * 2
        self.learner = nn.ModuleList()
        for i in range(numlayer-1):
            self.learner.append(
                LinearBlock(learner_in_features,
                            learner_in_features,
                            bias=True,
                            activation="leaky_relu",
                            norm=None,
                            negative_slope=negative_slope)
            )
        self.learner.append(
            LinearBlock(learner_in_features, 1, bias=True, activation=None, norm=None)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.extractor_1.reset_parameters()
        self.extractor_2.reset_parameters()
        for i in range(self.numlayer):
            self.learner[i].reset_parameters()

    def forward(self, x1, x2=None):
        x1 = self.extractor_1(x1)
        if x2 is None:
            x2 = torch.zeros_like(x1)
        else:
            x2 = self.extractor_2(x2)
        learner_x = torch.cat([x1, x2], dim=1)
        for i in range(self.numlayer):
            learner_x = self.learner[i](learner_x)
        weight = torch.sigmoid(learner_x)
        return weight


class MappingNN(nn.Module):
    def __init__(self, in_features: int, hid_features: int, out_features: int, numlayer: int, activation: str = None, norm: str = None, dropout: float = 0.5):
        super().__init__()
        if numlayer < 1:
            raise ValueError("numlayer cannot be less than 1")

        self.numlayer = numlayer
        self.dropout = dropout

        self.blocks = nn.ModuleList()
        if numlayer == 1:
            lin = LinearBlock(in_features, out_features, bias=True, activation=None, norm=None)
            self.blocks.append(lin)
        else:
            self.blocks.append(LinearBlock(in_features, hid_features, bias=True, activation=activation, norm=norm))
            for i in range(numlayer-2):
                lin = LinearBlock(hid_features, hid_features, bias=True, activation=activation, norm=norm)
                self.blocks.append(lin)
            self.blocks.append(LinearBlock(hid_features, out_features, bias=True, activation=None, norm=None))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.numlayer):
            self.blocks[i].reset_parameters()

    def forward(self, x):
        for i in range(self.numlayer):
            x = self.blocks[i](x)
            if i != self.numlayer-1:
                x = F.dropout(x, self.dropout, training=self.training)
        return x