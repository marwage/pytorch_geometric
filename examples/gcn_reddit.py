import os.path as osp
import argparse
import logging
import time
import subprocess

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(Net, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_channels, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(hidden_channels, num_classes, cached=True,
                             normalize=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def run():
    name = "gcn_reddit"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "Reddit")
    dataset = Reddit(path)
    data = dataset[0]

    timestamp_loading = time.time() - start
    logging.info("Loading data: %f", timestamp_loading)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    timestamp_data = time.time() - start
    logging.info("Copying data: %f", timestamp_data)    

    model = Net(dataset.num_features, dataset.num_classes, 512).to(device)
    
    timestamp_model = time.time() - start
    logging.info("Copying model: %f", timestamp_model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train(model, data, optimizer)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        logging.info(log.format(epoch, train_acc, best_val_acc, test_acc))

    timestamp_training = time.time() - start
    logging.info("Training: %f", timestamp_training)

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()