import os.path as osp
import time
import logging

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv

logging.basicConfig(filename='cluster_gcn_ppi.log',level=logging.DEBUG)
start = time.time()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

print('Partioning the graph... (this may take a while)')
num_parts = 1500
batch_size = 20
train_cluster_data = ClusterData(train_dataset, num_parts=num_parts, recursive=False,
                           save_dir=train_dataset.processed_dir)
train_loader = ClusterLoader(train_cluster_data, batch_size=batch_size, shuffle=True,
                       num_workers=0)
val_cluster_data = ClusterData(val_dataset, num_parts=num_parts, recursive=False,
                           save_dir=val_dataset.processed_dir)
val_loader = ClusterLoader(val_cluster_data, batch_size=batch_size, shuffle=True,
                       num_workers=0)
test_cluster_data = ClusterData(test_dataset, num_parts=num_parts, recursive=False,
                           save_dir=test_dataset.processed_dir)
test_loader = ClusterLoader(test_cluster_data, batch_size=batch_size, shuffle=True,
                       num_workers=0)

time_partioning_graph = time.time() - start
logging.info("partioning graph took " + str(time_partioning_graph))

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        hidden_channels = 1024
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_features, train_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = total_nodes = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.nll_loss(logits, data.y)
        loss.backward()
        optimizer.step()

        nodes = data.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test():
    model.eval()
    total_correct, total_nodes = 0, 0
    for data in test_loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        total_correct += (pred == data.y).sum().item()
        # total_nodes += data.x.sum().item()
        total_nodes += len(data)

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()


for epoch in range(1, 31):
    loss = train()
    accs = test()
    logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
          'Test: {:.4f}'.format(epoch, loss, *accs))

end = time.time()
time_whole_training = end - start
logging.info("whole training took " + str(time_whole_training))
