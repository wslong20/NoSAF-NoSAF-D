import os.path
import torch
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor, WikiCS, LINKXDataset


def cosine_distance(features: torch.Tensor):
    norms = torch.norm(features, p=2, dim=1, keepdim=True)
    normed_features = features / norms
    cosine_similarity_matrix = torch.mm(normed_features, normed_features.t())
    cosine_distance_matrix = 1 - cosine_similarity_matrix
    n = cosine_distance_matrix.size(0)
    avg_cosine_distance = cosine_distance_matrix.sum() / (n * n - n)
    return avg_cosine_distance.item()


def get_dataset(root: str, name: str):
    if not os.path.exists(root):
        os.mkdir(root)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name)
    elif name == 'CoraFull':
        dataset = CoraFull(root=root)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root=root, name=name)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root=root, name=name)
    elif name == 'WikiCS':
        dataset = WikiCS(root=root, is_undirected=True)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=root, name=name)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=root, name=name.lower())
    elif name == 'Actor':
        dataset = Actor(root=root)
    else:
        raise Exception('Unknown dataset.')

    return dataset


if __name__ == "__main__":
    from torch_geometric.utils import homophily
    from torch_geometric.utils import to_undirected
    import torch
    from typing import Union

    import torch
    from torch import Tensor

    from torch_geometric.typing import Adj, OptTensor, SparseTensor
    from torch_geometric.utils import degree, scatter
    import matplotlib.pyplot as plt

    homophily_root = "/home/ssd7T/remotewsl/GraphDataset/homophilic_graph_pyg"
    heterophily_root = "/home/ssd7T/remotewsl/GraphDataset/heterophilic_graph_pyg"
    data_cora = get_dataset(root=homophily_root, name="Cora")
    print(data_cora)
    data_citeseer = get_dataset(root=homophily_root, name="CiteSeer")
    data_pubmed = get_dataset(root=homophily_root, name="PubMed")
    data_corafull = get_dataset(root=homophily_root+"/corafull", name="CoraFull")
    data_computers = get_dataset(root=homophily_root, name="Computers")
    data_photo = get_dataset(root=homophily_root, name="Photo")
    data_cs = get_dataset(root=homophily_root, name="CS")
    data_physics = get_dataset(root=homophily_root, name="Physics")
    data_wikics = get_dataset(root=homophily_root, name="WikiCS")

    data_cornell = get_dataset(root=heterophily_root, name="Cornell")
    data_texas = get_dataset(root=heterophily_root, name="Texas")
    data_wisconsin = get_dataset(root=heterophily_root, name="Wisconsin")
    data_chameleon = get_dataset(root=heterophily_root, name="Chameleon")
    data_squirrel = get_dataset(root=heterophily_root, name="Squirrel")
    data_actor = get_dataset(root=heterophily_root, name="Actor")

    datas = [data_cora, data_citeseer, data_pubmed, data_corafull, data_computers, data_photo, data_cs, data_physics, data_wikics,
             data_cornell, data_texas, data_wisconsin, data_chameleon, data_squirrel, data_actor]
    data_name = ["Cora", "CiteSeer", "PubMed", "CoraFull", "Computers", "Photo", "CS", "Physics", "WikiCS", "Cornell", "Texas", "Wisconsin", "Chameleon", "Squirrel", "Actor"]
    for i in range(len(datas)):
        gd = datas[i].data
        gd.edge_index = to_undirected(gd.edge_index)
        graph_homophily = homophily(gd.edge_index, gd.y, method="node")
        print(data_name[i], gd)
        print(datas[i].num_classes)
        print(graph_homophily)
        print()

    # # data_ds = data_computers
    # data_ds = data_squirrel
    # # row, col = data_ds.data.edge_index
    # row, col = data_ds.data.edge_index
    # y = data_ds.data.y
    # out = torch.zeros(row.size(0), device=row.device)
    #
    # out[y[row] == y[col]] = 1.
    # out = scatter(out, col, 0, dim_size=y.size(0), reduce='mean')
    # print(out)
    # n, bins, patches = plt.hist(out, bins=5)
    # plt.show()
    # print(n)