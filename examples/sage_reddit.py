import os.path as osp
import time
import logging
import subprocess
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.conv = torch.nn.ModuleList()
        if num_layers < 1:
            raise Exception("You must have a least one layer")
        elif num_layers == 1:
            self.conv.append(SAGEConv(in_channels, out_channels, normalize=False))
        elif num_layers == 2:
            self.conv.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.conv.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        elif num_layers > 3:
            self.conv.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            for _ in range(1, num_layers-1):
                self.conv.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.conv.append(SAGEConv(hidden_channels, out_channels, normalize=False))


    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv[i](x, edge_index)
            if i != (self.num_layers - 1):
                x = F.relu(x)
        return F.log_softmax(x, dim=1)


def train(data, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
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
    name = "sage_reddit"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAGE(2, dataset.num_features, dataset.num_classes, 1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data = data.to(device)
    logging.debug("Type of data: " + str(type(data)))
    logging.debug("Attributes of data: " + str(dir(data)))
    logging.debug("Type of data.x: " + str(type(data.x)))
    logging.debug("Storage size of data.x: " + str(data.x.storage().size()))
    logging.debug("Type of data.edge_index: " + str(type(data.edge_index)))
    logging.debug("Storage size of data.edge_index: " + str(data.edge_index.storage().size()))

    time_stamp_data = time.time() - start
    logging.info("Copying data: " + str(time_stamp_data))
    
    model = model.to(device)

    time_stamp_model = time.time() - start
    logging.info("Copying model: " + str(time_stamp_model))

    for epoch in range(1, 31):
        loss = train(data, model, optimizer)
        # accs = test(data, model)
        # logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
        #     'Test: {:.4f}'.format(epoch, loss, *accs))

    time_stamp_training = time.time() - start
    logging.info("Training: " + str(time_stamp_training))

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()