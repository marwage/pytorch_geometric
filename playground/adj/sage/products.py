import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset

def load_data(transform=None):
    root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "products")
    dataset = PygNodePropPredDataset("ogbn-products", root, transform=transform)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    return data, split_idx, dataset.num_features, dataset.num_classes