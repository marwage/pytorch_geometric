import logging
import torch
import torch.nn.functional as F

# from torch_geometric.nn import SAGEConv
from .sage_conv import SAGEConv
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
        empty_cache = False
        dropout_prob = 0.2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = torch.zeros((1,), device=device)

        num_chunks = len(x_chunks)
        dropout_chunks = []

        # compute dropout for each chunk
        for x_chunk in x_chunks:
            x_chunk = x_chunk.to(device)
            dropout_chunk = F.dropout(x_chunk, p=dropout_prob, training=self.training)
            dropout_chunk = dropout_chunk.cpu()
            if empty_cache: torch.cuda.empty_cache()
            dropout_chunks.append(dropout_chunk)
        
        for i, layer in enumerate(self.conv):
            # load input for aggregation
            dropout = torch.cat(dropout_chunks)
            dropout = dropout.to(device)            

            # aggreate neighbourhood features
            aggr_chunks = []
            for adj_chunk in adj_chunks:
                adj_chunk = adj_chunk.to(device)
                aggr_chunk = matmul(adj_chunk, dropout, reduce="mean")
                aggr_chunk = aggr_chunk.cpu()
                if empty_cache: torch.cuda.empty_cache()
                aggr_chunks.append(aggr_chunk)
            dropout = dropout.cpu()
            if empty_cache: torch.cuda.empty_cache()

            # apply linear and (ReLU or (softmax and loss))
            new_dropout_chunks = []
            for j, (aggr_chunk, dropout_chunk) in enumerate(zip(aggr_chunks, dropout_chunks)):
                # apply linear
                aggr_chunk = aggr_chunk.to(device)
                dropout_chunk = dropout_chunk.to(device)
                linear_neighbours = layer.lin_l(aggr_chunk)
                aggr_chunk = aggr_chunk.cpu()
                if empty_cache: torch.cuda.empty_cache()
                linear_self = layer.lin_r(dropout_chunk)
                dropout_chunk = dropout_chunk.cpu()
                if empty_cache: torch.cuda.empty_cache()
                linear = linear_neighbours + linear_self
                linear_neighbours = linear_neighbours.cpu()
                linear_self = linear_self.cpu()
                if empty_cache: torch.cuda.empty_cache()

                # apply (ReLU or (softmax and loss))
                if i != (self.num_layers - 1):
                    relu_chunk = F.relu(linear)
                    linear = linear.cpu()
                    if empty_cache: torch.cuda.empty_cache()
                    dropout_chunk = F.dropout(relu_chunk, p=dropout_prob, training=self.training)
                    relu_chunk = relu_chunk.cpu()
                    dropout_chunk = dropout_chunk.cpu()
                    if empty_cache: torch.cuda.empty_cache()
                    new_dropout_chunks.append(dropout_chunk)
                    if j == (num_chunks - 1):
                        dropout_chunks = new_dropout_chunks
                else: # softmax and loss
                    softmax_chunk = F.log_softmax(linear, dim=1)
                    linear = linear.cpu()
                    if empty_cache: torch.cuda.empty_cache()
                    y_chunk = y_chunks[j].to(device)
                    train_mask_chunk = train_mask_chunks[j].to(device)
                    loss += F.nll_loss(softmax_chunk[train_mask_chunk], y_chunk[train_mask_chunk])
                    softmax_chunk = softmax_chunk.cpu()
                    y_chunk = y_chunk.cpu()
                    train_mask_chunk = train_mask_chunk.cpu() 
                    if empty_cache: torch.cuda.empty_cache()
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
