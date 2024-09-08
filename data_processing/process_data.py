import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from collections import defaultdict

def getcross_validation_split(local_dataset, n_folds=2, batch_size=1):
    # local_dataset should be an instance of a Dataset or a processed list of data objects
    train_ids, test_ids, valid_ids = split_ids(rnd_state.permutation(len(local_dataset)), folds=n_folds)
    splits = []

    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:
            gdata = [local_dataset[i] for i in split[fold_id]]
            loader = DataLoader(gdata,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
            loaders.append(loader)
        splits.append(loaders)

    return splits  # 0-train, 1-test, 2-valid

dir = './raw/weights_zip_'
adjacency_path = dir + 'A.txt'
graph_labels_path = dir + 'graph_labels.txt'
node_labels_path = dir + 'node_labels.txt'
edge_labels_path = dir + 'edge_labels.txt'
graph_indicators_path = dir + 'graph_indicator.txt'

edges = np.loadtxt(adjacency_path, delimiter=',', dtype=int)
graph_labels = np.loadtxt(graph_labels_path, delimiter=',', dtype=float)
node_labels = np.loadtxt(node_labels_path, dtype=int)
edge_labels = np.loadtxt(edge_labels_path, dtype=float)
graph_indicators = np.loadtxt(graph_indicators_path, dtype=int)-1

# Process the Data into `Data` Objects
graphs = defaultdict(lambda: {"nodes": [], "edges": [], "edge_labels": []})

# Organize nodes by graphs
for node_id, graph_id in enumerate(graph_indicators):
    graphs[graph_id]["nodes"].append(node_id)

# Organize edges by graphs
for (node1, node2), edge_label in zip(edges, edge_labels):
    graph_id = graph_indicators[node1]  # Assuming nodes are 1-based in graph_indicator
    graphs[graph_id]["edges"].append((node1, node2))
    graphs[graph_id]["edge_labels"].append(edge_label)

# Create Data objects
data_list = []

for graph_id, graph_data in graphs.items():
    edge_index = torch.tensor(graph_data["edges"], dtype=torch.long).t().contiguous()
    x = torch.tensor([node_labels[node] for node in graph_data["nodes"]], dtype=torch.float)
    edge_attr = torch.tensor(graph_data["edge_labels"], dtype=torch.float)
    
    data = Data(x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                y=torch.tensor(np.array(graph_labels[graph_id]), dtype=torch.long))
    data_list.append(data)

# Step 3: Use the Data in a `Dataset`
# Optionally, create a custom Dataset class
class MyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

# Example usage
local_dataset = MyGraphDataset(data_list)
loader = DataLoader(local_dataset, batch_size=32, shuffle=True)

# Now `loader` is ready to be used in your training loop
for data in loader:
    # Example: print batch data
    print(data)
