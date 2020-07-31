import os.path as osp
import time
import logging
import subprocess
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

from benchmarking.model.sage import sage
from benchmarking.dataset import reddit, flickr
from benchmarking.log import mw as mw_logging


def run():
    graph_dataset = "reddit"
    gnn_model = "sage"
    adj_matrix = True
    
    if not adj_matrix:
        name = "{}_{}_edge_index".format(gnn_model, graph_dataset)
    else:
        name = "{}_{}".format(gnn_model, graph_dataset)
    
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    if adj_matrix:
        transform = T.ToSparseTensor()
    else:
        transform = None
    if graph_dataset == "flickr":
        data, num_features, num_classes = flickr.load_data(transform)
    elif graph_dataset == "reddit":
        data, num_features, num_classes = reddit.load_data(transform)

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_hidden_layers = 2
    num_hidden_channels = 512
    model = sage.SAGE(num_hidden_layers, num_features, num_classes, num_hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data = data.to(device)
    time_stamp_data = time.time() - start
    logging.info("Copying data: " + str(time_stamp_data))

    logging.debug("---------- data ----------")
    logging.debug("Type of data: " + str(type(data)))
    logging.debug("Number of classes {}".format(num_classes))
    logging.debug("Number of edges {}".format(data.num_edges))
    for attribute in dir(data):
        if type(data[attribute]) is torch.Tensor:
            mw_logging.log_tensor(data[attribute], attribute)
        if type(data[attribute]) is SparseTensor:
            storage = data[attribute].storage
            mw_logging.log_sparse_storage(storage, attribute)
            
    mw_logging.log_peak_increase("After data.to(device)")

    model = model.to(device)

    time_stamp_model = time.time() - start
    logging.info("Copying model: " + str(time_stamp_model))

    logging.debug("-------- model ---------")
    logging.debug("Type of model: {}".format(type(model)))
    for i, param in enumerate(model.parameters()):
        mw_logging.log_tensor(param, "param {}".format(i))
    mw_logging.log_peak_increase("After model.to(device)")

    logging.debug("Type of optimizer: {}".format(type(optimizer)))
    logging.debug("Attributes of optimizer: {}".format(dir(optimizer)))

    # num_epochs = 30
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        loss = sage.train(data, data.train_mask, model, optimizer)
        accs = sage.test(data, model)
        logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
            'Test: {:.4f}'.format(epoch, loss, *accs))
        logging.info("Epoch: {:02d}, Loss: {:.4f}".format(epoch, loss))

    time_stamp_training = time.time() - start
    logging.info("Training: " + str(time_stamp_training))
    mw_logging.log_gpu_memory("End of training")

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()
