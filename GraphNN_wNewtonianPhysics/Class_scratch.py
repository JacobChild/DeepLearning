#Class_scratch.py
#combo of example from pygeometry and Dr. Ning's code

#%% Packages
from torch_geometric.datasets import Planetoid 
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.nn as pyg_nn
import torch 
import torch.nn as nn
from torch.functional import F
# %% Import data 
# check if already downloaded 
dataset = Planetoid(root='Data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0] 
#%% explore the data
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'data.x.shape: {data.x.shape}')
print(f'data.y.shape: {data.y.shape}')

# %% Visualize (the code below is AI recommended and runs, but takes a while)
# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.Graph()
# G.add_nodes_from(range(data.num_nodes))
# edges = data.edge_index.t().tolist()
# G.add_edges_from(edges)
# plt.figure(figsize=(10,10))
# nx.draw(G, node_size=10)
# plt.title('Cora dataset')
# plt.show()

# %% Make the nn.Module using GCNConv 
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, nhidden=16):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, nhidden)
        self.conv2 = pyg_nn.GCNConv(nhidden, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x #torch.log_softmax(x, dim=1)
    
def train(graphs, model, lossfn, optimizer):
    model.train()
    optimizer.zero_grad()
    
    yhat = model(graphs)
    loss = lossfn(yhat[graphs.train_mask], graphs.y[graphs.train_mask])
    
    loss.backward()
    optimizer.step()
    return loss

def test(graphs, model, lossfn): 
    model.eval() 
    
    with torch.no_grad():
        yhat = model(graphs)
        ytest = graphs.y[graphs.test_mask]
        loss = lossfn(yhat[graphs.test_mask], ytest)
        _, pred = yhat.max(dim=1)
        correct = float(pred.eq(ytest).sum().item())
        acc = correct / graphs.test_mask.sum().item()
        return acc, loss
    

# %% Instantiate the model
model = GCN(data.num_features, 16, dataset.num_classes, nhidden=16)
print(model)
# %% Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lossfn = nn.CrossEntropyLoss()
train_losses = []
test_losses = []
test_accs = []
epochs = 200
for epoch in range(epochs):
    loss = train(data, model, lossfn, optimizer)
    train_losses.append(loss.item())
    acc, test_loss = test(data, model, lossfn)
    test_losses.append(test_loss.item())
    test_accs.append(acc.item())
    # print(f'Epoch {epoch+1}/{epochs} loss: {loss.item()}')
#%% Plot the train_losses
import matplotlib.pyplot as plt
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
#Plot the test_losses
plt.plot(test_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.show()
#Plot the test_accs
plt.plot(test_accs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
    

# %%
