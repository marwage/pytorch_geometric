import os.path as osp
import time
import logging
import subprocess
import pdb
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.nn import SAGEConv


def log(start, when):
        mib = pow(2, 20)
        logging.debug("{:.1f}s:{}:active {:.2f}MiB, allocated {:.2f}MiB, reserved {:.2f}MiB".format(time.time() - start, when, torch.cuda.memory_stats()["active_bytes.all.allocated"] / mib, torch.cuda.memory_allocated() / mib, torch.cuda.memory_reserved() / mib))


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


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.conv):
            x = F.dropout(x, p=0.2, training=self.training)
            x = layer(x, edge_index)
            if i != (self.num_layers - 1):
                x = F.relu(x)
        return F.log_softmax(x, dim=1)


def train(data, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    log(-1, "After forward")
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    nodes = data.train_mask.sum().item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(data, model):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

    masks = [data.train_mask, data.val_mask, data.test_mask]
    for i, mask in enumerate(masks):
        total_correct[i] = (pred[mask] == data.y[mask]).sum().item()
        total_nodes[i] = mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()


def run():
    name = "sage_flickr"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
    dataset = Flickr(path)
    data = dataset[0]

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_hidden_layers = 2
    model = SAGE(num_hidden_layers, dataset.num_features, dataset.num_classes, 1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data = data.to(device)
    time_stamp_data = time.time() - start
    logging.info("Copying data: " + str(time_stamp_data))

    logging.debug("Type of data: " + str(type(data)))
    logging.debug("Number of classes {}".format(dataset.num_classes))
    for attribute in dir(data):
        if type(data[attribute]) is torch.Tensor:
            logging.debug("Shape of parameter: {}".format(data[attribute].size()))
            logging.debug("Data type of parameter: {}".format(data[attribute].dtype))
            logging.debug("Storage of parameter: {}".format(data[attribute].storage().size()))
    log(start, "After data.to(device)")

    model = model.to(device)

    time_stamp_model = time.time() - start
    logging.info("Copying model: " + str(time_stamp_model))

    logging.debug("Type of model: {}".format(type(model)))
    logging.debug("Model parameter sizes: {}".format([param.size() for param in model.parameters()]))
    for param in model.parameters():
        logging.debug("Shape of model parameter {}: {}".format(str(param.names), param.size()))
        logging.debug("Data type of data.{}: {}".format(str(param.names), param.dtype))
        logging.debug("Storage of data.{}: {}".format(str(param.names), param.storage().size()))
    log(start, "After model.to(device)")

    logging.debug("Type of optimizer: {}".format(type(optimizer)))
    logging.debug("Attributes of optimizer: {}".format(dir(optimizer)))

    for epoch in range(1, 31):
        loss = train(data, model, optimizer)
        accs = test(data, model)
        logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
            'Test: {:.4f}'.format(epoch, loss, *accs))

    time_stamp_training = time.time() - start
    logging.info("Training: " + str(time_stamp_training))

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()
