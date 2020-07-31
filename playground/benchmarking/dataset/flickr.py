import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr

def load_data(transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Flickr")
    dataset = Flickr(path, transform=transform)
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes
