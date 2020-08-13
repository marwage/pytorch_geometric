import logging
import torch
from torch.nn import Linear
import torch.nn.functional as F
from benchmarking.log import mw as mw_logging
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
        logging.debug("---------- SIGN.forward ----------")
        # mw_logging.log_peak_increase("After xs")
        hs = []
        # for i, x_i in enumerate(xs):
        #     if type(x_i) is torch.Tensor:
        #         mw_logging.log_tensor(x_i, "x_{}".format(i))
        for i, lin in enumerate(self.lins):
            logging.debug("---------- layer.forward ----------")
            # linear = lin(xs[i])
            linear = checkpoint(lin, xs[i])
            # mw_logging.log_peak_increase("After linear")
            # mw_logging.log_tensor(linear, "linear")
            mw_logging.log_timestamp("after linear")
            rlu = F.relu(linear)
            # mw_logging.log_peak_increase("After relu")
            # mw_logging.log_tensor(rlu, "relu")
            mw_logging.log_timestamp("after relu")
            dropo = F.dropout(rlu, p=0.2, training=self.training)
            # mw_logging.log_peak_increase("After dropout")
            # mw_logging.log_tensor(dropo, "dropo")
            mw_logging.log_timestamp("after dropout")
            hs.append(dropo)
        x = torch.cat(hs, dim=-1)
        # mw_logging.log_peak_increase("After concatenate xs")
        # mw_logging.log_tensor(x, "concatenate xs")
        mw_logging.log_timestamp("after cat")
        x = self.lin(x)
        # x = checkpoint(self.lin, x)
        # mw_logging.log_peak_increase("After lin")
        # mw_logging.log_tensor(x, "lin")
        mw_logging.log_timestamp("after lin")
        soft = F.log_softmax(x, dim=-1)
        # mw_logging.log_peak_increase("After softmax")
        # mw_logging.log_tensor(soft, "softmax")
        mw_logging.log_timestamp("after softmax")
        return soft


def train(xs, y, train_mask, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    # mw_logging.log_peak_increase("Before zero_grad")
    optimizer.zero_grad()
    # mw_logging.log_peak_increase("After zero_grad")
    logits = model(xs)
    # mw_logging.log_tensor(logits, "logits")
    # mw_logging.log_peak_increase("After forward")
    loss = F.nll_loss(logits[train_mask], y[train_mask])
    # mw_logging.log_peak_increase("After loss")
    mw_logging.log_timestamp("after forward")
    loss.backward()
    # mw_logging.log_peak_increase("After backward")
    # for i, param in enumerate(model.parameters()):
    #     mw_logging.log_tensor(param, "param {}".format(i))
    mw_logging.log_timestamp("after backward")
    optimizer.step()
    # mw_logging.log_peak_increase("After step")
    mw_logging.log_timestamp("after step")

    nodes = train_mask.sum().item()
    total_loss = loss.item() * nodes
    total_nodes = nodes

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
