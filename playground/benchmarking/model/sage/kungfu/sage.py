import logging
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from benchmarking.log import mw as mw_logging
from torch_sparse import matmul

from kungfu.torch.ops import all_gather


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

    def all_gather_padding(self, x, default_chunk_size, chunk_sizes_diff):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if x.size(0) < default_chunk_size:
            pad = torch.zeros((chunk_sizes_diff, x.size(1)), device=device)
            x_padded = torch.cat([x, pad])
            x_all = all_gather(x_padded)
        else:
            x_all = all_gather(x)
        size = x_all.size()
        x_all = torch.reshape(x_all, (size[0] * size[1], size[2]))
        if chunk_sizes_diff > 0:
            x_all = x_all[0:-chunk_sizes_diff]

        return x_all


    def forward(self, x, adj, default_chunk_size, chunk_sizes_diff):
        dropout_prob = 0.2

        act = x
        for i, layer in enumerate(self.conv):
            act_dropout = F.dropout(act, p=dropout_prob, training=self.training)

            act_dropout_all = self.all_gather_padding(act_dropout, default_chunk_size, chunk_sizes_diff)

            act_matmul = matmul(adj, act_dropout_all, reduce="mean")
            act_layer_l = layer.lin_l(act_matmul)
            act_layer_r = layer.lin_r(act_dropout)
            act_layer = act_layer_l + act_layer_r
            
            if i != (self.num_layers - 1):
                act_relu = F.relu(act_layer)
            else:
                act_relu = act_layer

            act = act_relu

        act_all = self.all_gather_padding(act, default_chunk_size, chunk_sizes_diff)
        softmax = F.log_softmax(act_all, dim=1)

        return softmax


def train(x, adj, y, train_mask, model, optimizer, default_chunk_size, chunk_sizes_diff):
    model.train()
    total_loss = total_nodes = 0
    optimizer.zero_grad()
    logits = model(x, adj, default_chunk_size, chunk_sizes_diff)
    loss = F.nll_loss(logits[train_mask], y[train_mask])
    # loss.backward() # TODO RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    # optimizer.step()

    nodes = train_mask.sum().item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(x, adj, y, model, masks):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    logits = model(x, adj)
    pred = logits.argmax(dim=1)

    for i, mask in enumerate(masks):
        total_correct[i] = (pred[mask] == y[mask]).sum().item()
        total_nodes[i] = mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()
