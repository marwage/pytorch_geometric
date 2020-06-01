import os.path as osp
import time
import logging

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

logging.basicConfig(filename='sign_ppi.log',level=logging.DEBUG)
start = time.time()


K = 3
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
transform = T.Compose([T.NormalizeFeatures(), T.SIGN(K)])
dataset = PPI(path, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

time_preprocessing_data = time.time() - start
logging.info("preprocessing data took " + str(time_preprocessing_data))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        num_hidden_channels = 512
        for _ in range(K + 1):
            self.lins.append(Linear(dataset.num_node_features, num_hidden_channels))
        self.lin = Linear((K + 1) * num_hidden_channels, dataset.num_classes)

    def forward(self, data):
        xs = [data.x] + [data[f'x{i}'] for i in range(1, K + 1)]
        for i, lin in enumerate(self.lins):
            out = F.dropout(F.relu(lin(xs[i])), p=0.5, training=self.training)
            xs[i] = out
        x = torch.cat(xs, dim=-1)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# start_copying = time.time()
model = Net().to(device)
# time_copying = time.time() - start_copying
# logging.info("copying took " + str(time_copying))

loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for data in loader:
        num_graphs = data.num_graphs
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    loss = train()
    log = 'Epoch: {:03d}, Train: {:.4f}'
    logging.info(log.format(epoch, loss))
    # train_acc, val_acc, tmp_test_acc = test()
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # logging.info(log.format(epoch, train_acc, best_val_acc, test_acc))

end = time.time()
time_whole_training = end - start
logging.info("whole training took " + str(time_whole_training))
