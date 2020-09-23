import argparse
import time
import logging
import subprocess
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

from benchmarking.model.sage import sage
from benchmarking.model.sign import sign
from benchmarking.dataset import reddit, flickr, products
from benchmarking.log import mw as mw_logging


def run(graph_dataset, gnn_model, adj_matrix):
    if gnn_model == "sage" and not adj_matrix:
        name = "{}_{}_edge_index".format(gnn_model, graph_dataset)
    else:
        name = "{}_{}".format(gnn_model, graph_dataset)
    
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    mw_logging.set_start()

    transform_list = []
    if gnn_model == "sign":
        k = 3
        transform_list.append(T.NormalizeFeatures())
        transform_list.append(T.SIGN(k))
    if adj_matrix:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gnn_model == "sage":
        data = data.to(device)
    elif gnn_model == "sign":
        xs = [data.x.to(device)] + [data[f'x{i}'].to(device) for i in range(1, k + 1)]
        y = data.y.to(device)

    num_hidden_channels = 512
    if gnn_model == "sage":
        num_hidden_layers = 2
        model = sage.SAGE(num_hidden_layers, num_features, num_classes, num_hidden_channels)
    elif gnn_model == "sign":
        model = sign.SIGN(k, num_features, num_classes, num_hidden_channels)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    if graph_dataset == "products":
        masks = [split_idx["train"], split_idx["valid"], split_idx["test"]]
        train_mask = split_idx["train"]
    else:
        masks = [data.train_mask, data.val_mask, data.test_mask]
        train_mask = data.train_mask

    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        if gnn_model == "sage":
            loss = sage.train(data, train_mask, model, optimizer)
            # accs = sage.test(data, model, masks)
        elif gnn_model == "sign":
            loss = sign.train(xs, y, train_mask, model, optimizer)
            accs = sign.test(xs, y, masks, model)
        # logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
        #     'Test: {:.4f}'.format(epoch, loss, *accs))
        logging.info("Epoch: {:02d}, Loss: {:.4f}".format(epoch, loss))

    mw_logging.log_timestamp("finish training")
    mw_logging.log_gpu_memory("End of training")

    monitoring_gpu.terminate()


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
    run(args.dataset, args.model, args.adj)
