"""Example using GNNExplainer
"""

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GNNExplainer, global_mean_pool
from torch_geometric.data import DataLoader

dataset = 'MUTAG'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TUDataset')
dataset = TUDataset(path, dataset, transform=T.NormalizeFeatures())
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


model = Net(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 201):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

single_graph = dataset[5]   # get first graph object
print(single_graph)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {single_graph.num_nodes}')
print(f'Number of edges: {single_graph.num_edges}')
print(f'Average node degree: {single_graph.num_edges / single_graph.num_nodes:.2f}')
print(f'Contains isolated nodes: {single_graph.contains_isolated_nodes()}')
print(f'Contains self-loops: {single_graph.contains_self_loops()}')
print(f'Is undirected: {single_graph.is_undirected()}')


single_graph = single_graph.to(device)
x, edge_index = single_graph.x, single_graph.edge_index
# print(x, edge_index)


explainer = GNNExplainer(model, epochs=200)
node_idx = 0

# need to do the union of all the nodes to get full graph explanation...
node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index, batch=torch.zeros(single_graph.num_nodes, dtype=torch.int64))
ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, threshold=0.5) #, y=single_graph.y)
print(node_feat_mask)
print(edge_mask)
plt.show()


# 
# """
# Cora - node classification
# """

# import os.path as osp

# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv, GNNExplainer

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

# print(f'Dataset: {dataset}:')
# print('====================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')


# data = dataset[0]

# print('=============================================================')

# # Gather some statistics about the first graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
# print(f'Contains self-loops: {data.contains_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_features, 16)
#         self.conv2 = GCNConv(16, 16)
#         self.conv3 = GCNConv(16, dataset.num_classes)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# data = data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# x, edge_index = data.x, data.edge_index

# for epoch in range(1, 201):
#     model.train()
#     optimizer.zero_grad()
#     log_logits = model(x, edge_index)
#     loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()

# explainer = GNNExplainer(model, epochs=200)
# node_idx = 10
# node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
# ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y, threshold=0.8)
# print(max(edge_mask))
# plt.show()