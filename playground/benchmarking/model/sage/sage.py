import logging
import torch
import torch.nn.functional as F

# from torch_geometric.nn import SAGEConv
from .sage_conv import SAGEConv
from benchmarking.log import mw as mw_logging


class SAGE(torch.nn.Module):
    def __init__(self, num_hidden_layers, in_channels, out_channels, hidden_channels):
        super(SAGE, self).__init__()
        self.num_layers = num_hidden_layers + 1
        self.conv = torch.nn.ModuleList()
        if num_hidden_layers < 1:
            raise Exception("You must have at least one hidden layer")
        else:
            self.conv.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            for _ in range(1, num_hidden_layers):
                self.conv.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.conv.append(SAGEConv(hidden_channels, out_channels, normalize=False))


    def forward(self, x, edge_index):
        # logging.debug("---------- SAGE.forward ----------")
        # mw_logging.log_tensor(x, "x in")
        for i, layer in enumerate(self.conv):
            # logging.debug("---------- layer forward ----------")
            x = F.dropout(x, p=0.2, training=self.training)
            # mw_logging.log_tensor(x, "after dropout")
            # mw_logging.log_peak_increase("after dropout")
            # mw_logging.log_timestamp("after dropout")
            x = layer(x, edge_index)
            # mw_logging.log_tensor(x, "after layer")
            # mw_logging.log_peak_increase("after layer")
            # mw_logging.log_timestamp("after layer")
            if i != (self.num_layers - 1):
                x = F.relu(x)
                # mw_logging.log_tensor(x, "after relu")
                # mw_logging.log_peak_increase("after relu")
                # mw_logging.log_timestamp("after relu")
        softmax = F.log_softmax(x, dim=1)
        # mw_logging.log_tensor(softmax, "after softmax")
        # mw_logging.log_peak_increase("after softmax")
        # mw_logging.log_timestamp("after softmax")
        return softmax


def train(data, train_mask, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    # mw_logging.log_peak_increase("Before zero_grad")
    optimizer.zero_grad()
    # mw_logging.log_peak_increase("After zero_grad")
    if hasattr(data, "adj_t"):
        logits = model(data.x, data.adj_t)
    else:
        logits = model(data.x, data.edge_index)
    # mw_logging.log_tensor(logits, "logits")
    # mw_logging.log_peak_increase("After forward")
    loss = F.nll_loss(logits[train_mask], data.y[train_mask])
    # mw_logging.log_peak_increase("After loss")
    # mw_logging.log_timestamp("after forward")
    loss.backward()
    # mw_logging.log_peak_increase("After backward")
    # mw_logging.log_timestamp("after backward")
    optimizer.step()
    # mw_logging.log_peak_increase("After step")
    # mw_logging.log_timestamp("after step")

    nodes = data.train_mask.sum().item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(data, model, masks):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    if hasattr(data, "adj_t"):
        logits = model(data.x, data.adj_t)
    else:
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

    for i, mask in enumerate(masks):
        total_correct[i] = (pred[mask] == data.y[mask]).sum().item()
        total_nodes[i] = mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()
