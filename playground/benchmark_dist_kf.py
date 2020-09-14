import argparse
import time
import logging
import subprocess
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

from benchmarking.model.sage.kungfu import sage
from benchmarking.dataset import reddit, flickr, products
from benchmarking.log import mw as mw_logging

from kungfu.python import current_rank, current_cluster_size


def run(graph_dataset):
    name = "{}_{}_chunk_act".format("sage", graph_dataset)
    
    # monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    mw_logging.set_start()

    torch.autograd.set_detect_anomaly(True)

    transform_list = []
    transform_list.append(T.ToSparseTensor())
    
    if not transform_list:
        transform = None
    else:
        transform = T.Compose(transform_list)

    if graph_dataset == "flickr":
        data, num_features, num_classes = flickr.load_data(transform)
    elif graph_dataset == "reddit":
        data, num_features, num_classes = reddit.load_data(transform)
    elif graph_dataset == "products":
        data, split_idx, num_features, num_classes = products.load_data(transform)
    else:
        raise Exception("Not a valid dataset")

    rank = current_rank()
    cluster_size = current_cluster_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_hidden_channels = 512
    num_hidden_layers = 2
    model = sage.SAGE(num_hidden_layers, num_features, num_classes, num_hidden_channels)

    model.to(device)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def backward_hook(module, grad_input, grad_output):
        [mw_logging.log_tensor(t, "grad_input") for t in grad_input]
        [mw_logging.log_tensor(t, "grad_output") for t in grad_output]

    if graph_dataset == "products":
        masks = [split_idx["train"], split_idx["valid"], split_idx["test"]]
        train_mask = split_idx["train"]
    else:
        masks = [data.train_mask, data.val_mask, data.test_mask]
        train_mask = data.train_mask

    x_chunk = data.x.chunk(cluster_size)[rank]
    x = x_chunk.to(device)
    y = data.y.to(device)
    adj = data.adj_t
    chunk_size = x_chunk.size(0)
    l = rank * chunk_size
    if l + chunk_size <= adj.size(0):
        u = l + chunk_size
    else:
        u = adj.size(0)
    adj_chunk = adj[l:u]
    adj = adj_chunk.to(device)
    train_mask = train_mask.to(device) # train_mask is not split if all workers calculate the softmax and loss

    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        loss = sage.train(x, adj, y, train_mask, model, optimizer)
        # accs = sage.test(x, adj, y, model, masks)
        # logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
        #     'Test: {:.4f}'.format(epoch, loss, *accs))
        logging.info("Epoch: {:02d}, Loss: {:.4f}".format(epoch, loss))

    mw_logging.log_timestamp("End of training")
    mw_logging.log_gpu_memory("End of training")

    # monitoring_gpu.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking GNNs")
    parser.add_argument("--model",
                        type=str,
                        default="sage",
                        help="sage or sign")
    parser.add_argument("--dataset",
                        type=str,
                        default="reddit",
                        help="flickr, reddit or products")
    parser.add_argument("--adj",
                        action="store_true")
    args = parser.parse_args()
    run(args.dataset)
