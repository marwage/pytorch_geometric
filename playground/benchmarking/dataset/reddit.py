import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit

def load_data(transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Reddit")
    dataset = Reddit(path, transform=transform)
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes