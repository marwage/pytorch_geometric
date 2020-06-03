import os.path as osp
import time
import logging

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score

logging.basicConfig(filename='cluster_gcn_reddit_f1.log',level=logging.DEBUG)
start = time.time()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

print('Partioning the graph... (this may take a while)')
cluster_data = ClusterData(data, num_parts=1500, recursive=False,
                           save_dir=dataset.processed_dir)
loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True,
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
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = total_nodes = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        nodes = data.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test():
    model.eval()
    
    ys, preds = [[], [], []], [[], [], []]
    for data in loader:
        logits = model(data.x.to(device), data.edge_index.to(device))
        pred = logits.argmax(dim=1)
        masks = [data.train_mask, data.val_mask, data.test_mask]
        for _, mask in enumerate(masks):
            preds.append((pred[mask] > 0).float().cpu())
            ys.append(data.y[mask])
    
    f1_scores = []
    for y, pred in zip(ys, preds):
        y, pred = torch.cat(y, dim=0).numpy(), torch.cat(pred, dim=0).numpy()
        f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        f1_scores.append(f1)

    return f1_scores

for epoch in range(1, 31):
    loss = train()
    accs = test()
    logging.info('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
          'Test: {:.4f}'.format(epoch, loss, *accs))

end = time.time()
time_whole_training = end - start
logging.info("whole training took " + str(time_whole_training))
