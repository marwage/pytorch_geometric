import os.path as osp
import time
import logging
import subprocess
import pdb
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from torch_sparse import SparseTensor
import mw_logging


class SIGN(torch.nn.Module):
    def __init__(self, k, in_channels, out_channels, hidden_channels):
        super(SIGN, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.k = k

        for _ in range(self.k + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
        self.lin = Linear((self.k + 1) * hidden_channels, out_channels)

    def forward(self, xs):
        logging.debug("---------- SIGN.forward ----------")
        # mw_logging.log_peak_increase("After xs")
        hs = []
        for i, x_i in enumerate(xs):
            if type(x_i) is torch.Tensor:
                mw_logging.log_tensor(x_i, "x_{}".format(i))
        for i, lin in enumerate(self.lins):
            logging.debug("---------- layer.forward ----------")
            linear = lin(xs[i])
            # mw_logging.log_peak_increase("After linear")
            mw_logging.log_tensor(linear, "linear")
            rlu = F.relu(linear)
            # mw_logging.log_peak_increase("After relu")
            mw_logging.log_tensor(rlu, "relu")
            dropo = F.dropout(rlu, p=0.2, training=self.training)
            # mw_logging.log_peak_increase("After dropout")
            mw_logging.log_tensor(dropo, "dropo")
            hs.append(dropo)
        x = torch.cat(hs, dim=-1)
        # mw_logging.log_peak_increase("After concatenate xs")
        mw_logging.log_tensor(x, "concatenate xs")
        x = self.lin(x)
        # mw_logging.log_peak_increase("After lin")
        mw_logging.log_tensor(x, "lin")
        soft = F.log_softmax(x, dim=-1)
        # mw_logging.log_peak_increase("After softmax")
        mw_logging.log_tensor(soft, "softmax")
        return soft


def train(xs, y, train_idx, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    mw_logging.log_peak_increase("Before zero_grad")
    optimizer.zero_grad()
    mw_logging.log_peak_increase("After zero_grad")
    logits = model(xs)
    mw_logging.log_tensor(logits, "logits")
    mw_logging.log_peak_increase("After forward")
    loss = F.nll_loss(logits[train_idx], y[train_idx])
    mw_logging.log_peak_increase("After loss")
    loss.backward()
    mw_logging.log_peak_increase("After backward")
    for i, param in enumerate(model.parameters()):
        mw_logging.log_tensor(param, "param {}".format(i))
    optimizer.step()
    mw_logging.log_peak_increase("After step")

    nodes = train_idx.size(0).item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(data, model):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    logits = model(data)
    pred = logits.argmax(dim=1)

    masks = [data.train_mask, data.val_mask, data.test_mask]
    for i, mask in enumerate(masks):
        total_correct[i] = (pred[mask] == data.y[mask]).sum().item()
        total_nodes[i] = mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()


def run():
    name = "sign_products"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    k = 3
    root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
    # transform = T.Compose([T.ToSparseTensor(), T.NormalizeFeatures(), T.SIGN(k)])
    transform = T.Compose([T.NormalizeFeatures(), T.SIGN(k)])
    dataset = PygNodePropPredDataset('ogbn-products', root, transform=transform)
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    logging.debug("Split indices: {}".format(split_idx))
    train_idx = split_idx['train']

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_hidden_channels = 512
    model = SIGN(k, dataset.num_features, dataset.num_classes, num_hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    xs = [data.x.to(device)] + [data[f'x{i}'].to(device) for i in range(1, k + 1)]
    y = data.y.to(device)
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
        loss = train(xs, y, train_idx, model, optimizer)
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
