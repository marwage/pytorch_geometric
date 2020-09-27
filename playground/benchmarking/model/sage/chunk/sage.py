import logging
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from benchmarking.log import mw as mw_logging
from torch_sparse import matmul


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

    def forward(self, x_chunks, adj_chunks, y_chunks, train_mask_chunks):
        dropout_prob = 0.2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = torch.zeros((1,), device=device)
        num_chunks = len(x_chunks)

        # compute dropout for each chunk
        dropout_chunks = []
        for i, x_chunk in enumerate(x_chunks):
            x_chunk = x_chunk.to(device)
            dropout_chunk = F.dropout(x_chunk, p=dropout_prob, training=self.training)
            del x_chunk
            dropout_chunk = dropout_chunk.cpu()
            dropout_chunks.append(dropout_chunk)
        
        for i, layer in enumerate(self.conv):
            # layer to gpu
            layer = layer.to(device)

            # load input for aggregation
            dropout_mat = torch.cat(dropout_chunks)
            dropout_mat = dropout_mat.to(device)

            # apply linear and ((ReLU and dropout) or (softmax and loss))
            for j, adj_chunk in enumerate(adj_chunks):
                adj_chunk = adj_chunk.to(device) 
                # graph convolution
                conv = layer(dropout_mat, adj_chunk) # TODO not possible
                del adj_chunk

                # apply ((ReLU and dropout) or (softmax and loss))
                if i != (self.num_layers - 1): # (ReLU and dropout)
                    relu_chunk = F.relu(conv)
                    mw_logging.log_current_active("Before conv to CPU {}-{}".format(i, j))
                    conv = conv.cpu() # TODO no decrease
                    mw_logging.log_current_active("After conv to CPU {}-{}".format(i, j))

                    dropout_chunk = F.dropout(relu_chunk, p=dropout_prob, training=self.training)
                    relu_chunk = relu_chunk.cpu()
                    dropout_chunk = dropout_chunk.cpu()
                    dropout_chunks[j] = dropout_chunk

                else: # softmax and loss
                    softmax_chunk = F.log_softmax(conv, dim=1)
                    mw_logging.log_current_active("Before conv to CPU {}-{}".format(i, j))
                    conv = conv.cpu() # TODO no decrease
                    mw_logging.log_current_active("After conv to CPU {}-{}".format(i, j))
                    y_chunk = y_chunks[j].to(device)
                    train_mask_chunk = train_mask_chunks[j].to(device)
                    loss += F.nll_loss(softmax_chunk[train_mask_chunk], y_chunk[train_mask_chunk])
                    mw_logging.log_current_active("Before softmax_chunk to CPU {}-{}".format(i, j))
                    softmax_chunk = softmax_chunk.cpu() # TODO no decrease; small could be neglected
                    mw_logging.log_current_active("After softmax_chunk to CPU {}-{}".format(i, j))
                    del y_chunk
                    mw_logging.log_current_active("Before delete train_mask_chunk {}-{}".format(i, j))
                    del train_mask_chunk # TODO no decrease; small could be neglected
                    mw_logging.log_current_active("After delete train_mask_chunk {}-{}".format(i, j))

            # layer to cpu
            layer = layer.cpu()

        return loss / num_chunks


def train(x, adj, y, train_mask, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = model(x, adj, y, train_mask)
    # loss.backward()
    # optimizer.step()

    return loss.item()


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