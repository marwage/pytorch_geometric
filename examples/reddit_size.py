import os.path as osp
import time
import logging
import subprocess
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv




def run():
    name = "reddit_size"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    time_stamp_copying = time.time() - start
    logging.info("Copying data model:" + str(time_stamp_copying))

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()