import os.path as osp
import time
import logging
import subprocess
import pdb
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
# from torch_geometric.nn import SAGEConv

from sage_conv_checkpoint import SAGEConv
from torch_sparse import SparseTensor
import mw_logging
from torch.utils.checkpoint import checkpoint


class SAGE(torch.nn.Module):
    def __init__(self, num_hidden_layers, in_channels, out_channels, hidden_channels):
        super(SAGE, self).__init__()
        self.num_layers = num_hidden_layers + 1
        self.conv = torch.nn.ModuleList()
        if num_hidden_layers < 1:
            raise Exception("You must have at least one hidden layer")
        elif num_hidden_layers == 1:
            self.conv.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.conv.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        elif num_hidden_layers > 1:
            self.conv.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            for _ in range(num_hidden_layers - 1):
                self.conv.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.conv.append(SAGEConv(hidden_channels, out_channels, normalize=False))


    def one_layer(self, layer, i, x, edge_index):
        logging.debug("---------- layer forward ----------")
        h = F.dropout(x, p=0.2, training=self.training)
        h = layer(h, edge_index)
        if i != (self.num_layers - 1):
            h = F.relu(h)
        return h


    def forward(self, x, edge_index):
        logging.debug("---------- SAGE.forward ----------")
        h = x
        for i, layer in enumerate(self.conv):
            # h = self.one_layer(layer, i, h, edge_index)
            h = checkpoint(self.one_layer, layer, i, h, edge_index)
        softmax = F.log_softmax(h, dim=1)
        return softmax


def train(data, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    mw_logging.log_peak_increase("Before zero_grad")
    optimizer.zero_grad()
    mw_logging.log_peak_increase("After zero_grad")
    logits = model(data.x, data.adj_t)
    mw_logging.log_tensor(logits, "logits")
    mw_logging.log_peak_increase("After forward")
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    mw_logging.log_peak_increase("After loss")
    loss.backward()
    mw_logging.log_peak_increase("After backward")
    for i, param in enumerate(model.parameters()):
        mw_logging.log_tensor(param, "param {}".format(i))
    optimizer.step()
    mw_logging.log_peak_increase("After step")

    nodes = data.train_mask.sum().item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(data, model):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    logits = model(data.x, data.adj_t)
    pred = logits.argmax(dim=1)

    masks = [data.train_mask, data.val_mask, data.test_mask]
    for i, mask in enumerate(masks):
        total_correct[i] = (pred[mask] == data.y[mask]).sum().item()
        total_nodes[i] = mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()


def run():
    name = "sage_reddit_checkpoint"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path, transform=T.ToSparseTensor())
    data = dataset[0]

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_hidden_layers = 2
    num_hidden_channels = 512
    model = SAGE(num_hidden_layers, dataset.num_features, dataset.num_classes, num_hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data = data.to(device)
    time_stamp_data = time.time() - start
    logging.info("Copying data: " + str(time_stamp_data))

    logging.debug("---------- data ----------")
    logging.debug("Type of data: " + str(type(data)))
    logging.debug("Number of classes {}".format(dataset.num_classes))
    logging.debug("Number of edges {}".format(data.num_edges))
    for attribute in dir(data):
        if type(data[attribute]) is torch.Tensor:
            mw_logging.log_tensor(data[attribute], attribute)
        if type(data[attribute]) is SparseTensor:
            storage = data[attribute].storage
            mw_logging.log_sparse_storage(storage, attribute)
            
    mw_logging.log_peak_increase("After data.to(device)")

    model = model.to(device)

    time_stamp_model = time.time() - start
    logging.info("Copying model: " + str(time_stamp_model))

    logging.debug("-------- model ---------")
    logging.debug("Type of model: {}".format(type(model)))
    for i, param in enumerate(model.parameters()):
        mw_logging.log_tensor(param, "param {}".format(i))
    mw_logging.log_peak_increase("After model.to(device)")

    logging.debug("Type of optimizer: {}".format(type(optimizer)))
    logging.debug("Attributes of optimizer: {}".format(dir(optimizer)))

    # num_epochs = 30
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        loss = train(data, model, optimizer)
        accs = test(data, model)
        logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
            'Test: {:.4f}'.format(epoch, loss, *accs))
        # logging.info("Epoch: {:02d}, Loss: {:.4f}".format(epoch, loss))

    time_stamp_training = time.time() - start
    logging.info("Training: " + str(time_stamp_training))
    mw_logging.log_gpu_memory("End of training")

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()
