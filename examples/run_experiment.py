import time
import logging
import torch
import subprocess

import cluster_gcn_reddit_callable
name = "cluster_gcn_reddit"

monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])

logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
start = time.time()

# preprocessing
num_partitions = 300
logging.info("Number of partitions: " + str(num_partitions))
batch_size = 10
logging.info("Batch size: " + str(batch_size))
loader, dataset_info = cluster_gcn_reddit_callable.preprocessing(num_partitions, batch_size)

time_stamp_preprocessing = time.time() - start
logging.info("Preprocessing:" + str(time_stamp_preprocessing))

# create
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_hidden_channels = 512
logging.info("Number of hidden channels: " + str(num_hidden_channels))
model = cluster_gcn_reddit_callable.Net(dataset_info["num_features"], dataset_info["num_classes"], num_hidden_channels).to(device)
learning_rate = 0.01
logging.info("Learning rate: " + str(learning_rate))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

time_stamp_model = time.time() - start
logging.info("Model:" + str(time_stamp_model))

# training
num_epochs = 30
logging.info("Number of epochs: " + str(num_epochs))
log = cluster_gcn_reddit_callable.run(loader, model, optimizer, device, num_epochs)
logging.info(log)

time_stamp_training = time.time() - start
logging.info("Training:" + str(time_stamp_training))

monitoring_gpu.terminate()
