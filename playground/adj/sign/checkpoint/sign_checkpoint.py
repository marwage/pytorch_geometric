import torch
from torch.nn import Linear
import torch.nn.functional as F

import logging
import mw_logging
from torch.utils.checkpoint import checkpoint


class SIGN(torch.nn.Module):
    def __init__(self, k, in_channels, out_channels, hidden_channels):
        super(SIGN, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.k = k

        for _ in range(self.k + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
        self.lin = Linear((self.k + 1) * hidden_channels, out_channels)

    def forward(self, xs):
        hs = []
        for i, lin_i in enumerate(self.lins):
            # linear = lin_i(xs[i])
            linear = checkpoint(lin_i, xs[i])
            mw_logging.log_peak_increase("After linear {}".format(i))
            rlu = F.relu(linear)
            mw_logging.log_peak_increase("After relu {}".format(i))
            dropo = F.dropout(rlu, p=0.2, training=self.training)
            mw_logging.log_peak_increase("After dropout {}".format(i))
            hs.append(dropo)
        x = torch.cat(hs, dim=-1)
        mw_logging.log_peak_increase("After concatenation")
        x = self.lin(x)
        mw_logging.log_peak_increase("After final linear")
        # x = checkpoint(self.lin, x)
        soft = F.log_softmax(x, dim=-1)
        mw_logging.log_peak_increase("After softmax")
        return soft


def train(xs, y, train_mask, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    optimizer.zero_grad()
    mw_logging.log_peak_increase("Before forward")
    logits = model(xs)
    mw_logging.log_peak_increase("After forward")
    logging.debug("gradient function: {}".format(logits.grad_fn))
    loss = F.nll_loss(logits[train_mask], y[train_mask])
    loss.backward()
    mw_logging.log_peak_increase("After backward")
    optimizer.step()

    nodes = train_mask.sum().item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

    mw_logging.log_gpu_memory("End of train")

    return total_loss / total_nodes


@torch.no_grad()
def test(xs, y, masks, model):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    logits = model(xs)
    pred = logits.argmax(dim=1)

    for i, mask in enumerate(masks):
        total_correct[i] = (pred[mask] == y[mask]).sum().item()
        total_nodes[i] = mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()
