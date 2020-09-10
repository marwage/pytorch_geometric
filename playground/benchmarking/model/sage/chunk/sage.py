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


    def forward(self, x, adj):
        dropout_prob = 0.2
        chunk_size = 2 ** 11
        keep_all_tensors = False
        log_tensors = False
        log_memory = False
        swap_to_cpu = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if keep_all_tensors: all_tensors = []
        if keep_all_tensors: all_tensors.append(("x", x))
        if keep_all_tensors: all_tensors.append(("adj", adj))

        a = x
        if log_tensors: mw_logging.log_tensor(a, "x")
        if log_memory: mw_logging.log_gpu_memory("x")
        for i, layer in enumerate(self.conv):
            a = a.to(device)
            if log_tensors: mw_logging.log_tensor(a, "a_{}".format(i))
            if log_memory: mw_logging.log_gpu_memory("a_{}".format(i))
            if keep_all_tensors: all_tensors.append(("a_{}".format(i), a))
            a_dropout = F.dropout(a, p=dropout_prob, training=self.training)
            if log_tensors: mw_logging.log_tensor(a_dropout, "a_dropout_{}".format(i))
            if log_memory: mw_logging.log_gpu_memory("a_dropout_{}".format(i))
            if keep_all_tensors: all_tensors.append(("a_dropout_{}".format(i), a))
            if swap_to_cpu:
                a_cpu = a.cpu()
                if log_tensors: mw_logging.log_tensor(a_cpu, "a_cpu_{}".format(i))
                if log_memory: mw_logging.log_gpu_memory("a_cpu_{}".format(i))
                if keep_all_tensors: all_tensors.append(("a_cpu_{}".format(i), a_cpu))

            relus = []
            a_chunks = torch.split(a_dropout, chunk_size)
            if log_tensors: [mw_logging.log_tensor(chunk, "a_chunk_{}_{}".format(i, j)) for j, chunk in enumerate(a_chunks)]
            if log_memory: mw_logging.log_gpu_memory("a_chunks_{}".format(i))
            # adj_chunks = torch.split(adj, chunk_size) # SparseTensor has no attribute split
            for chunk in range(len(a_chunks)):
                if keep_all_tensors: all_tensors.append(("a_chunk_{}_{}".format(i, chunk), a_chunks[chunk]))
                l = chunk * chunk_size
                if l + chunk_size <= adj.size(0):
                    u = l + chunk_size
                else:
                    u = adj.size(0)
                adj_chunk = adj[l:u]

                a_matmul = matmul(adj_chunk, a_dropout, reduce="mean")
                if log_tensors: mw_logging.log_tensor(a_matmul, "a_matmul_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_matmul_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_matmul_{}_{}".format(i, chunk), a_matmul))
                a_l = layer.lin_l(a_matmul)
                if log_tensors: mw_logging.log_tensor(a_l, "a_l_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_l_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_l_{}_{}".format(i, chunk), a_l))
                a_matmul_cpu = a_matmul.cpu()
                if log_tensors: mw_logging.log_tensor(a_matmul_cpu, "a_matmul_cpu_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_matmul_cpu_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_matmul_cpu_{}_{}".format(i, chunk), a_matmul_cpu))
                a_r = layer.lin_r(a_chunks[chunk])
                if log_tensors: mw_logging.log_tensor(a_r, "a_r_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_r_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_r_{}_{}".format(i, chunk), a_r))
                a_layer = a_l + a_r
                if log_tensors: mw_logging.log_tensor(a_layer, "a_layer_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_layer_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_layer_{}_{}".format(i, chunk), a_layer))
                a_l_cpu = a_l.cpu()
                if log_tensors: mw_logging.log_tensor(a_l_cpu, "a_l_cpu_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_l_cpu_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_l_cpu_{}_{}".format(i, chunk), a_l_cpu))
                a_r_cpu = a_r.cpu()
                if log_tensors: mw_logging.log_tensor(a_r_cpu, "a_r_cpu_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_r_cpu_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_r_cpu_{}_{}".format(i, chunk), a_r_cpu))

                if log_tensors: mw_logging.log_tensor(a_layer, "a_layer_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_layer_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_layer_{}_{}".format(i, chunk), a_layer))
                if i != (self.num_layers - 1):
                    a_relu = F.relu(a_layer)
                    if log_tensors: mw_logging.log_tensor(a_relu, "a_relu_{}_{}".format(i, chunk))
                    if log_memory: mw_logging.log_gpu_memory("a_relu_{}_{}".format(i, chunk))
                    if keep_all_tensors: all_tensors.append(("a_relu_{}_{}".format(i, chunk), a_relu))
                    if swap_to_cpu:
                        a_layer_cpu = a_layer.cpu()
                        if log_tensors: mw_logging.log_tensor(a_layer_cpu, "a_layer_cpu_{}_{}".format(i, chunk))
                        if log_memory: mw_logging.log_gpu_memory("a_layer_cpu_{}_{}".format(i, chunk))
                        if keep_all_tensors: all_tensors.append(("a_layer_cpu_{}_{}".format(i, chunk), a_layer_cpu))
                else:
                    a_relu = a_layer
                if swap_to_cpu:
                    a_relu_cpu = a_relu.cpu()
                    if log_tensors: mw_logging.log_tensor(a_relu_cpu, "a_relu_cpu_{}_{}".format(i, chunk))
                    if log_memory: mw_logging.log_gpu_memory("a_relu_cpu_{}_{}".format(i, chunk))
                    if keep_all_tensors: all_tensors.append(("a_relu_cpu_{}_{}".format(i, chunk), a_relu_cpu))
                    relus.append(a_relu_cpu)
                else:
                    relus.append(a_relu)
            if swap_to_cpu:
                a_dropout_cpu = a_dropout.cpu()
                if log_tensors: mw_logging.log_tensor(a_dropout_cpu, "a_dropout_cpu_{}_{}".format(i, chunk))
                if log_memory: mw_logging.log_gpu_memory("a_dropout_cpu_{}_{}".format(i, chunk))
                if keep_all_tensors: all_tensors.append(("a_dropout_cpu_{}_{}".format(i, chunk), a_dropout_cpu))
            a = torch.cat(relus)
            if log_tensors: mw_logging.log_tensor(a, "a_cat_{}_{}".format(i, chunk))
            if log_memory: mw_logging.log_gpu_memory("a_cat_{}_{}".format(i, chunk))
            if keep_all_tensors: all_tensors.append(("a_cat_{}_{}".format(i, chunk), a))

        a = a.to(device)
        if log_tensors: mw_logging.log_tensor(a, "a_before_softmax")
        if log_memory: mw_logging.log_gpu_memory("a_before_softmax")
        if keep_all_tensors: all_tensors.append(("a_before_softmax", a))
        softmax = F.log_softmax(a, dim=1)
        if log_tensors: mw_logging.log_tensor(softmax, "softmax")
        if log_memory: mw_logging.log_gpu_memory("softmax")
        if keep_all_tensors: all_tensors.append(("softmax", softmax))
        if swap_to_cpu:
            a_cpu = a.cpu()
            if log_tensors: mw_logging.log_tensor(a_cpu, "a_cpu_after_softmax")
            if log_memory: mw_logging.log_gpu_memory("a_cpu_after_softmax")
            if keep_all_tensors: all_tensors.append(("a_cpu_after_softmax", softmax))

        if keep_all_tensors: logging.debug("------ all tensors ------")
        if keep_all_tensors: [mw_logging.log_tensor(tensor, name) for name, tensor in all_tensors]

        return softmax


def train(x, adj, y, train_mask, model, optimizer):
    model.train()
    total_loss = total_nodes = 0
    optimizer.zero_grad()
    logits = model(x, adj)
    loss = F.nll_loss(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

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
