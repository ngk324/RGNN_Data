import torch
import torch.nn.functional as F
import warnings
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from collections import defaultdict

def pause():
    programPause = input("Press the <ENTER> key to continue...")

class MyOwnDataset(InMemoryDataset):
    
    num_node_labels = 3
    
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        # torch.serialization.add_safe_globals([MyOwnDataset, InMemoryDataset])
        # self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_node_labels = self.num_node_labels
        self.process()
        
    # def __num_node_labels__(self):
    #     return self.num_node_labels

    @property
    def raw_file_names(self):
        return ['A.txt', 'graph_labels.txt', 'node_labels.txt', 'edge_labels.txt', 'graph_indicators.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.process_()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def process_(self):
        dir = "../weights_zip/raw/weights_zip_"
        adjacency_path = dir + "A.txt"
        graph_labels_path = dir + "graph_labels.txt"
        node_labels_path = dir + "node_labels.txt"
        edge_labels_path = dir + "edge_labels.txt"
        graph_indicators_path = dir + "graph_indicator.txt"

        edges = np.loadtxt(adjacency_path, delimiter=",", dtype=int)
        graph_labels = np.loadtxt(graph_labels_path, delimiter=",", dtype=float)
        node_labels = np.loadtxt(node_labels_path, dtype=int)    
        edge_labels = np.loadtxt(edge_labels_path, dtype=float)
        graph_indicators = np.loadtxt(graph_indicators_path, dtype=int) - 1
            
        graphs = defaultdict(lambda: {"nodes": [], "edges": [], "edge_labels": []})

        for node_id, graph_id in enumerate(graph_indicators):
            graphs[graph_id]["nodes"].append(node_id)
        
        idx_dec = [0]
        for i in range(0, max(graphs.keys())):
            idx_dec.append(idx_dec[-1] + len(graphs[i]["nodes"]))

        for (node1, node2), edge_label in zip(edges, edge_labels):
            graph_id = graph_indicators[node1]
            graphs[graph_id]["edges"].append((node1, node2))
            graphs[graph_id]["edge_labels"].append(edge_label)
            
        for i in range(len(graphs)):
            new_edges = []
            for edge in graphs[i]["edges"]:
                edge_ = list(edge)
                edge_[0] -= idx_dec[i]
                edge_[1] -= idx_dec[i]
                new_edges.append(tuple(edge_))
            graphs[i]["edges"] = new_edges

        data_list = []

        for graph_id, graph_data in graphs.items():
            edge_index = torch.tensor(graph_data["edges"], dtype=torch.long).t().contiguous()
            x = torch.tensor([node_labels[node] for node in graph_data["nodes"]], dtype=torch.float).view(-1,1)
            x = F.one_hot(x.view(-1).long(), num_classes=self.num_node_labels).float()
            edge_attr = torch.tensor(graph_data["edge_labels"], dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(np.array(graph_labels[graph_id]), dtype=torch.float).view(1,5),
            )
            
            # print(f"shape of data.x is {data.x.shape}")
            # print(f"shape of data.y is {data.y.shape}")
            
            data_list.append(data)
            
        return data_list
