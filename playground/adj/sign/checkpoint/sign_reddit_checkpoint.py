import time
import logging
import subprocess
import torch
from torch.nn import Linear
import torch_geometric.transforms as T
import sign_checkpoint
import reddit
from torch_sparse import SparseTensor
import mw_logging


def run():
    name = "sign_reddit_checkpoint_lin_i"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
    start = time.time()

    k = 3
    transform = T.Compose([T.NormalizeFeatures(), T.SIGN(k)])
    data, num_features, num_classes = reddit.load_data(transform)

    time_stamp_preprocessing = time.time() - start
    logging.info("Loading data: " + str(time_stamp_preprocessing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_hidden_channels = 512
    model = sign_checkpoint.SIGN(k, num_features, num_classes, num_hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    xs = [data.x.to(device)] + [data[f'x{i}'].to(device) for i in range(1, k + 1)]
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    masks = [train_mask, val_mask, test_mask]
    time_stamp_data = time.time() - start
    logging.info("Copying data: " + str(time_stamp_data))
    logging.debug("---------- data ----------")
    logging.debug("Type of data: " + str(type(data)))
    logging.debug("Number of classes {}".format(dataset.num_classes))
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
    num_epochs = 2
    for epoch in range(1, num_epochs + 1):
        loss = sign_checkpoint.train(xs, y, train_mask, model, optimizer)
        accs = sign_checkpoint.test(xs, y, masks, model)
        logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
            'Test: {:.4f}'.format(epoch, loss, *accs))
        logging.info("Epoch: {:02d}, Loss: {:.4f}".format(epoch, loss))

    time_stamp_training = time.time() - start
    logging.info("Training: " + str(time_stamp_training))
    mw_logging.log_gpu_memory("End of training")

    monitoring_gpu.terminate()


if __name__ == "__main__":
    run()
