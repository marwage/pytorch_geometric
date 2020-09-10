import argparse
import time
import logging
import subprocess
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

from benchmarking.model.sage.chunk import sage
from benchmarking.dataset import reddit, flickr, products
from benchmarking.log import mw as mw_logging


def run(graph_dataset):
    name = "{}_{}_chunk_act".format("sage", graph_dataset)
    
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
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

    # mw_logging.log_timestamp("preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logging.debug("---------- data ----------")
    # logging.debug("Type of data: " + str(type(data)))
    # logging.debug("Number of classes {}".format(num_classes))
    # logging.debug("Number of edges {}".format(data.num_edges))
    # for attribute in dir(data):
    #     if type(data[attribute]) is torch.Tensor:
    #         mw_logging.log_tensor(data[attribute], attribute)
    #     if type(data[attribute]) is SparseTensor:
    #         storage = data[attribute].storage
    #         mw_logging.log_sparse_storage(storage, attribute)
            
    # mw_logging.log_peak_increase("After data.to(device)")

    num_hidden_channels = 512
    num_hidden_layers = 2
    model = sage.SAGE(num_hidden_layers, num_features, num_classes, num_hidden_channels)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    def backward_hook(module, grad_input, grad_output):
        [mw_logging.log_tensor(t, "grad_input") for t in grad_input]
        [mw_logging.log_tensor(t, "grad_output") for t in grad_output]
    
    # model.register_backward_hook(backward_hook)

    # mw_logging.log_timestamp("copying model")

    # logging.debug("-------- model ---------")
    # logging.debug("Type of model: {}".format(type(model)))
    # for i, param in enumerate(model.parameters()):
    #     mw_logging.log_tensor(param, "param {}".format(i))
    # mw_logging.log_peak_increase("After model.to(device)")

    # logging.debug("Type of optimizer: {}".format(type(optimizer)))
    # logging.debug("Attributes of optimizer: {}".format(dir(optimizer)))

    if graph_dataset == "products":
        masks = [split_idx["train"], split_idx["valid"], split_idx["test"]]
        train_mask = split_idx["train"]
    else:
        masks = [data.train_mask, data.val_mask, data.test_mask]
        train_mask = data.train_mask

    x = data.x
    y = data.y.to(device)
    adj = data.adj_t.to(device)
    train_mask.to(device)

    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        loss = sage.train(x, adj, y, train_mask, model, optimizer)
        accs = sage.test(x, adj, y, model, masks)
        logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
            'Test: {:.4f}'.format(epoch, loss, *accs))
        # logging.info("Epoch: {:02d}, Loss: {:.4f}".format(epoch, loss))

    mw_logging.log_timestamp("End of training")
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
    run(args.dataset)
