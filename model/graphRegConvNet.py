import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.graph_conv import GraphConv
from torch_scatter import scatter_add, scatter_mean, scatter_max

def pause():
    input("Paused. Press Any Key to Continue...")

def scatter_(name, src, index, dim_size=None):
    assert name in ["add", "mean", "max"]
    if name == "add":
        assert index.max() < src.size(0), "Index out of bounds for scatter_add"
        return scatter_add(src, index, dim=0, dim_size=dim_size)
    elif name == "mean":
        assert index.max() < src.size(0), "Index out of bounds for scatter_mean"
        return scatter_mean(src, index, dim=0, dim_size=dim_size)
    elif name == "max":
        fill_value = -1e9
        assert index.max() < src.size(0), "Index out of bounds for scatter_max"
        out, _ = scatter_max(src, index, dim=0, dim_size=dim_size)
        out[out == fill_value] = 0
        return out


class GraphRegConv_GNN(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, out_dim, drop_prob=0.5, max_k=3, device=None
    ):
        super(GraphRegConv_GNN, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv1 = GraphConv(in_channels, out_channels)        
        # self.conv2 = GraphConv(out_channels, out_channels)
        
        # print(self.in_channels)
        # print(self.out_channels)
        # pause()
        
        self.conv1 = GraphConv(in_channels, out_channels)        
        self.conv2 = GraphConv(out_channels, out_channels) 
        
        self.out_dim = out_dim
        self.max_k = max_k

        # self.layer_norm = torch.nn.BatchNorm1d(self.out_channels)
        
        self.layer_norm = torch.nn.BatchNorm1d(self.out_channels)
        self.dropout = torch.nn.Dropout(p=drop_prob)

        # self.bn_out = torch.nn.BatchNorm1d(self.out_channels * 3 * max_k)
        
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * 3 * max_k) 
        self.out_fun = torch.nn.Identity()

        # self.lin1 = torch.nn.Linear(
        #     self.out_channels * 3 * max_k, self.out_channels * 2
        # )
        
        self.lin1 = torch.nn.Linear(
            self.out_channels * 3 * max_k, self.out_channels * 2
        )
        self.lin2 = torch.nn.Linear(self.out_channels * 2, self.out_channels)
        self.lin3 = torch.nn.Linear(self.out_channels, self.out_dim)
        
        self.pre_train = False
        self.reset_parameters()

    def reset_parameters(self):
        print("reset parameters")
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.layer_norm.reset_parameters()
        self.bn_out.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        torch.nn.init.xavier_uniform_(self.conv1.lin_rel.weight)
        torch.nn.init.xavier_uniform_(self.conv2.lin_rel.weight)

    def forward(self, data, hidden_layer_aggregator=None):
        
        l = data.x
        edge_index = data.edge_index
        
        # print(l)
        # print(edge_index)
        # pause()
        
        k = self.max_k
        size = data.batch.max().item() + 1
        
        h_i = F.leaky_relu(self.conv1(l, edge_index).to(self.device))
        h_i = self.layer_norm(h_i)
        
        h_graph_mean = scatter_("mean", h_i, data.batch, dim_size=size)
        h_graph_max = scatter_("max", h_i, data.batch, dim_size=size)
        h_graph_sum = scatter_("add", h_i, data.batch, dim_size=size)
        H = [torch.cat([h_graph_mean, h_graph_max, h_graph_sum], 1)]

        for i in range(k - 1):
            h_i = F.leaky_relu(self.conv2(h_i, edge_index))
            h_i = self.layer_norm(h_i)
            
            h_graph_mean = scatter_("mean", h_i, data.batch, dim_size=size)
            h_graph_max = scatter_("max", h_i, data.batch, dim_size=size)
            h_graph_sum = scatter_("add", h_i, data.batch, dim_size=size)

            h_graph = torch.cat((h_graph_mean, h_graph_max, h_graph_sum), 1)            
            H.append(h_graph)

        h_k = torch.cat(H, dim=1)
        x = self.bn_out(h_k)
        
        # x = F.leaky_relu(self.lin1(x))
        
        x = F.relu(self.lin1(x))
        
        x = self.dropout(x)
        # x = F.leaky_relu(self.lin2(x))
        
        x = F.relu(self.lin2(x))
        
        x = self.dropout(x)
        x = self.out_fun(self.lin3(x))
        # x = F.relu(self.lin3(x))

        x_validity = x
        
        return x, x_validity
