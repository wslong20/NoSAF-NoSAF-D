import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GCNBlock, NodeWeightLearner, MappingNN


class NoSAF(nn.Module):
    def __init__(self, in_features: int, gnn_in_features: int, out_features: int,
                 input_map_numlayer: int, backbone_numlayer: int, output_map_numlayer: int,
                 input_map_norm: str = "batchnorm", backbone_norm: str = "batchnorm", output_map_norm: str = "none",
                 learner_hid_features: int = 4, learner_numlayer: int = 2, negative_slope: float = 0.2,
                 add_self_loops: bool = True, normalize: bool = True, dropout: float = 0.5):
        super().__init__()
        if input_map_numlayer < 1 or backbone_numlayer < 1 or output_map_numlayer < 1:
            raise ValueError("numlayer cannot be less than 1")
        self.input_map = MappingNN(
            in_features=in_features,
            hid_features=gnn_in_features * 2,
            out_features=gnn_in_features,
            numlayer=input_map_numlayer,
            activation="relu",
            norm=input_map_norm,
            dropout=dropout
        )

        # backbone
        self.backbone_numlayer = backbone_numlayer
        self.backbone = nn.ModuleList()
        self.learners = nn.ModuleList()
        self.learners.append(
            NodeWeightLearner(in_features=gnn_in_features,
                              hid_features=learner_hid_features,
                              numlayer=learner_numlayer,
                              negative_slope=negative_slope)
        )  # input learner
        for i in range(backbone_numlayer):
            self.backbone.append(
                GCNBlock(
                    in_features=gnn_in_features,
                    out_features=gnn_in_features,
                    bias=True,
                    activation="relu",
                    norm=backbone_norm,
                    add_self_loops=add_self_loops,
                    normalize=normalize
                )
            )
            self.learners.append(
                NodeWeightLearner(
                    in_features=gnn_in_features,
                    hid_features=learner_hid_features,
                    numlayer=learner_numlayer,
                    negative_slope=negative_slope
                )
            )

        self.output_map = MappingNN(
            in_features=gnn_in_features,
            hid_features=gnn_in_features * 2,
            out_features=out_features,
            numlayer=output_map_numlayer,
            activation="relu",
            norm=output_map_norm,
            dropout=dropout
        )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.input_map.reset_parameters()
        self.learners[0].reset_parameters()
        for i in range(self.backbone_numlayer):
            self.backbone[i].reset_parameters()
            self.learners[i+1].reset_parameters()
        self.output_map.reset_parameters()

    def forward(self, x, adj):
        x = self.input_map(x)
        nw = self.learners[0](x)  # node weight
        x = x * nw
        fused_node_rep = x
        for i in range(self.backbone_numlayer):
            x = self.backbone[i](x, adj)
            nw = self.learners[i+1](x, fused_node_rep)
            x = x * nw
            fused_node_rep = fused_node_rep + x

            x = F.dropout(x, self.dropout, self.training)
        out = self.output_map(fused_node_rep)

        # out = F.log_softmax(out, dim=-1)
        return out


class DeepNoSAF(nn.Module):
    def __init__(self, in_features: int, gnn_in_features: int, out_features: int,
                 input_map_numlayer: int, backbone_numlayer: int, output_map_numlayer: int,
                 input_map_norm: str = "batchnorm", backbone_norm: str = "batchnorm", output_map_norm: str = "none",
                 learner_hid_features: int = 4, learner_numlayer: int = 2, negative_slope: float = 0.2,
                 add_self_loops: bool = True, normalize=True, dropout: float = 0.5):
        super().__init__()
        if input_map_numlayer < 1 or backbone_numlayer < 1 or output_map_numlayer < 1:
            raise ValueError("numlayer cannot be less than 1")
        self.input_map = MappingNN(
            in_features=in_features,
            hid_features=gnn_in_features * 2,
            out_features=gnn_in_features,
            numlayer=input_map_numlayer,
            activation="relu",
            norm=input_map_norm,
            dropout=dropout
        )

        # backbone
        self.backbone_numlayer = backbone_numlayer
        self.backbone = nn.ModuleList()
        self.learners = nn.ModuleList()
        self.learners.append(
            NodeWeightLearner(in_features=gnn_in_features,
                              hid_features=learner_hid_features,
                              numlayer=learner_numlayer,
                              negative_slope=negative_slope)
        )  # input learner
        for i in range(backbone_numlayer):
            self.backbone.append(
                GCNBlock(
                    in_features=gnn_in_features,
                    out_features=gnn_in_features,
                    bias=True,
                    activation="relu",
                    norm=backbone_norm,
                    add_self_loops=add_self_loops,
                    normalize=normalize
                )
            )
            self.learners.append(
                NodeWeightLearner(
                    in_features=gnn_in_features,
                    hid_features=learner_hid_features,
                    numlayer=learner_numlayer,
                    negative_slope=negative_slope
                )
            )

        self.output_map = MappingNN(
            in_features=gnn_in_features,
            hid_features=gnn_in_features * 2,
            out_features=out_features,
            numlayer=output_map_numlayer,
            activation="relu",
            norm=output_map_norm,
            dropout=dropout
        )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.input_map.reset_parameters()
        self.learners[0].reset_parameters()
        for i in range(self.backbone_numlayer):
            self.backbone[i].reset_parameters()
            self.learners[i+1].reset_parameters()
        self.output_map.reset_parameters()

    def forward(self, x, adj):
        x = self.input_map(x)
        nw = self.learners[0](x)  # node weight
        x = x * nw
        fused_node_rep = x
        for i in range(self.backbone_numlayer):
            x = self.backbone[i](x, adj)
            # x = checkpoint(self.backbone[i], x, adj)
            nw = self.learners[i+1](x, fused_node_rep)
            x_filtered = x * nw
            x = x_filtered + (1 - nw) * fused_node_rep
            fused_node_rep = fused_node_rep + x_filtered

            x = F.dropout(x, self.dropout, self.training)
        out = self.output_map(fused_node_rep)

        # out = F.log_softmax(out, dim=-1)
        return out