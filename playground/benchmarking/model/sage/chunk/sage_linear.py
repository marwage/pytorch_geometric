import logging
import torch
import torch.nn.functional as F
import math

from benchmarking.log import mw as mw_logging
from torch_sparse import matmul


class SAGE(torch.nn.Module):
    def __init__(self, num_hidden_layers, in_channels, out_channels, hidden_channels):
        super(SAGE, self).__init__()
        self.num_layers = num_hidden_layers + 1
        self.linear_layers = []
        if num_hidden_layers < 1:
            raise Exception("You must have at least one hidden layer")
        else:
            self.linear_layers.append(self.generate_linear(in_channels, hidden_channels))
            for _ in range(1, num_hidden_layers):
                self.linear_layers.append(self.generate_linear(hidden_channels, hidden_channels))
            self.linear_layers.append(self.generate_linear(hidden_channels, out_channels))

    def generate_linear(self, num_in, num_out):
        def generate_weights_bias(num_in, num_out):
            require_grad = True
            weights = torch.empty((num_in, num_out), requires_grad=require_grad)
            k = math.sqrt(1 / num_in)
            torch.nn.init.uniform_(weights, a=-k, b=k)
            bias = torch.empty((num_out,), requires_grad=require_grad)
            torch.nn.init.uniform_(bias, a=-k, b=k)
            return weights, bias
        return generate_weights_bias(num_in, num_out), generate_weights_bias(num_in, num_out)

    def parameters(self):
            params = []
            for linear_neigh, linear_self in self.linear_layers:
                weights, bias = linear_neigh
                params.append(weights)
                params.append(bias)
                weights, bias = linear_self
                params.append(weights)
                params.append(bias)
            return params

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
        
        for i, layer in enumerate(self.linear_layers):
            # layer to gpu
            linear_neigh, linear_self = layer
            linear_neigh_weights, linear_neigh_bias = linear_neigh
            linear_self_weights, linear_self_bias = linear_self
            linear_neigh_weights, linear_neigh_bias = linear_neigh_weights.to(device), linear_neigh_bias.to(device)
            linear_self_weights, linear_self_bias = linear_self_weights.to(device), linear_self_bias.to(device)

            # load input for aggregation
            dropout_mat = torch.cat(dropout_chunks)
            dropout_mat = dropout_mat.to(device)   

            # aggreate neighbourhood features
            aggr_chunks = []
            for j, adj_chunk in enumerate(adj_chunks):
                adj_chunk = adj_chunk.to(device)
                aggr_chunk = matmul(adj_chunk, dropout_mat, reduce="mean")
                del adj_chunk
                aggr_chunk = aggr_chunk.cpu()
                aggr_chunks.append(aggr_chunk)
            del dropout_mat

            # apply linear and ((ReLU and dropout) or (softmax and loss))
            for j, (aggr_chunk, dropout_chunk) in enumerate(zip(aggr_chunks, dropout_chunks)):
                # apply linear
                aggr_chunk_gpu = aggr_chunk.to(device)
                dropout_chunk_gpu = dropout_chunk.to(device)
                linear_neighbours = torch.add(torch.matmul(aggr_chunk_gpu, linear_neigh_weights), linear_neigh_bias)
                mw_logging.log_current_active("Before delete aggr_chunk_gpu {}-{}".format(i, j))
                del aggr_chunk_gpu # TODO no decrease
                mw_logging.log_current_active("After delete aggr_chunk_gpu {}-{}".format(i, j))
                linear_self = torch.add(torch.matmul(dropout_chunk_gpu, linear_self_weights), linear_self_bias)
                mw_logging.log_current_active("Before delete dropout_chunk_gpu {}-{}".format(i, j))
                del dropout_chunk_gpu # TODO no decrease
                mw_logging.log_current_active("After delete dropout_chunk_gpu {}-{}".format(i, j))
                linear = linear_neighbours + linear_self
                linear_neighbours = linear_neighbours.cpu()
                linear_self = linear_self.cpu()

                # apply ((ReLU and dropout) or (softmax and loss))
                if i != (self.num_layers - 1): # (ReLU and dropout)
                    relu_chunk = F.relu(linear)
                    mw_logging.log_current_active("Before linear to CPU {}-{}".format(i, j))
                    linear = linear.cpu() # TODO no decrease
                    mw_logging.log_current_active("After linear to CPU {}-{}".format(i, j))

                    dropout_chunk = F.dropout(relu_chunk, p=dropout_prob, training=self.training)
                    relu_chunk = relu_chunk.cpu()
                    dropout_chunk = dropout_chunk.cpu()
                    dropout_chunks[j] = dropout_chunk

                else: # softmax and loss
                    softmax_chunk = F.log_softmax(linear, dim=1)
                    mw_logging.log_current_active("Before linear to CPU {}-{}".format(i, j))
                    linear = linear.cpu() # TODO no decrease
                    mw_logging.log_current_active("After linear to CPU {}-{}".format(i, j))
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
            linear_neigh_weights, linear_neigh_bias = linear_neigh_weights.cpu(), linear_neigh_bias.cpu()
            linear_self_weights, linear_self_bias = linear_self_weights.cpu(), linear_self_bias.cpu()

        return loss / num_chunks


def train(x, adj, y, train_mask, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = model(x, adj, y, train_mask)
    loss.backward()
    optimizer.step()

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
