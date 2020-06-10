import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv


def preprocessing(num_partitions, batch_size):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]

    dataset_info = {"num_features": dataset.num_features, "num_classes": dataset.num_classes}

    cluster_data = ClusterData(data, num_parts=num_partitions, recursive=False,
                            save_dir=dataset.processed_dir)
    loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True,
                        num_workers=0)
    return loader, dataset_info


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(loader, model, optimizer, device):
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
def test(loader, model, device):
    model.eval()
    total_correct, total_nodes = [0, 0, 0], [0, 0, 0]
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        masks = [data.train_mask, data.val_mask, data.test_mask]
        for i, mask in enumerate(masks):
            total_correct[i] += (pred[mask] == data.y[mask]).sum().item()
            total_nodes[i] += mask.sum().item()

    return (torch.Tensor(total_correct) / torch.Tensor(total_nodes)).tolist()


def run(loader, model, optimizer, device, num_epochs):
    log = ""
    for epoch in range(1, num_epochs + 1):
        loss = train(loader, model, optimizer, device)
        accs = test(loader, model, device)
        log += "Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n".format(epoch, loss, *accs)
    return log


def main():
    num_partitions = 1500
    batch_size = 20
    loader, dataset_info = preprocessing(num_partitions, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(dataset_info["num_features"], dataset_info["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 30
    log = run(loader, model, optimizer, device, num_epochs)
    print(log)


if __name__ == "__main__":
    main()
