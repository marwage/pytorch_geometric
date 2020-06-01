import os.path as osp
import time
import logging

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T

logging.basicConfig(filename='sign_reddit.log',level=logging.DEBUG)
start = time.time()


K = 3
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
transform = T.Compose([T.NormalizeFeatures(), T.SIGN(K)])
train_dataset = PPI(path, transform=transform, split='train')
val_dataset = PPI(path, transform=transform, split='val')
test_dataset = PPI(path, transform=transform, split='test')
train_data = train_dataset[0]
val_data = val_dataset[0]
test_data = test_dataset[0]

time_preprocessing_data = time.time() - start
logging.info("preprocessing data took " + str(time_preprocessing_data))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        num_hidden_channels = 512
        for _ in range(K + 1):
            self.lins.append(Linear(train_dataset.num_node_features, num_hidden_channels))
        self.lin = Linear((K + 1) * num_hidden_channels, train_dataset.num_classes)

    def forward(self):
        xs = [data.x] + [data[f'x{i}'] for i in range(1, K + 1)]
        for i, lin in enumerate(self.lins):
            out = F.dropout(F.relu(lin(xs[i])), p=0.5, training=self.training)
            xs[i] = out
        x = torch.cat(xs, dim=-1)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_copying = time.time()
model, data = Net().to(device), data.to(device)
time_copying = time.time() - start_copying
logging.info("copying took " + str(time_copying))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_data], train_data.y).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for data in [train_data, val_data, test_data]:
        pred = logits[data].max(1)[1]
        acc = pred.eq(data.y).sum().item() / data.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    logging.info(log.format(epoch, train_acc, best_val_acc, test_acc))

end = time.time()
time_whole_training = end - start
logging.info("whole training took " + str(time_whole_training))
