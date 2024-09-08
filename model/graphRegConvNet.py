import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.graph_conv import GraphConv

# from torch_geometric.utils import torch_scatter
# from torch_geometric.utils import scatter
# torch.set_default_tensor_type(torch.DoubleTensor)

from torch_scatter import scatter_add, scatter_mean, scatter_max


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
        self.conv1 = GraphConv(in_channels, out_channels)
        self.conv2 = GraphConv(out_channels, out_channels)

        self.out_dim = out_dim
        self.max_k = max_k

        self.layer_norm = torch.nn.LayerNorm(self.out_channels)
        self.dropout = torch.nn.Dropout(p=drop_prob)

        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * 3 * max_k)
        self.out_fun = torch.nn.Identity()

        self.lin1 = torch.nn.Linear(
            self.out_channels * 3 * max_k, self.out_channels * 2
        )
        self.lin2 = torch.nn.Linear(self.out_channels * 2, self.out_channels)
        self.lin3 = torch.nn.Linear(self.out_channels, self.out_dim)
        
        # Predict validity of entry (whether it should be -1)
        self.conv1_ = GraphConv(in_channels, out_channels)
        self.conv2_ = GraphConv(out_channels, out_channels)
        self.lin1_ = torch.nn.Linear(
            self.out_channels * 3 * max_k, self.out_channels * 2
        )
        self.lin2_ = torch.nn.Linear(self.out_channels * 2, self.out_channels)
        self.validity_pred = torch.nn.Linear(self.out_channels, self.out_dim)
        
        self.pre_train = False
        self.reset_parameters()

    def reset_parameters(self):
        print("reset parameters")
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.layer_norm.reset_parameters()
        self.bn_out.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data, hidden_layer_aggregator=None):
        
        l = data.x
        edge_index = data.edge_index
        
        k = self.max_k
        size = data.batch.max().item() + 1
        
        h_i = self.conv1(l, edge_index).to(self.device)
        h_i = self.layer_norm(h_i)
        
        h_graph_mean = scatter_("mean", h_i, data.batch, dim_size=size)
        h_graph_max = scatter_("max", h_i, data.batch, dim_size=size)
        h_graph_sum = scatter_("add", h_i, data.batch, dim_size=size)
        H = [torch.cat([h_graph_mean, h_graph_max, h_graph_sum], 1)]
        
        # h_i_ = self.conv1_(l, edge_index).to(self.device)
        # h_i_ = self.layer_norm(h_i_)
        
        # h_graph_mean_ = scatter_("mean", h_i_, data.batch, dim_size=size)
        # h_graph_max_ = scatter_("max", h_i_, data.batch, dim_size=size)
        # h_graph_sum_ = scatter_("add", h_i_, data.batch, dim_size=size)
        # H_ = [torch.cat([h_graph_mean_, h_graph_max_, h_graph_sum_], 1)]

        for i in range(k - 1):
            h_i = self.conv2(h_i, edge_index)
            h_i = self.layer_norm(h_i)
            
            h_graph_mean = scatter_("mean", h_i, data.batch, dim_size=size)
            h_graph_max = scatter_("max", h_i, data.batch, dim_size=size)
            h_graph_sum = scatter_("add", h_i, data.batch, dim_size=size)

            h_graph = torch.cat((h_graph_mean, h_graph_max, h_graph_sum), 1)            
            H.append(h_graph)
            
            # h_i_ = self.conv2_(h_i_, edge_index)
            # h_i_ = self.layer_norm(h_i_)
            
            # h_graph_mean_ = scatter_("mean", h_i_, data.batch, dim_size=size)
            # h_graph_max_ = scatter_("max", h_i_, data.batch, dim_size=size)
            # h_graph_sum_ = scatter_("add", h_i_, data.batch, dim_size=size)

            # h_graph_ = torch.cat((h_graph_mean_, h_graph_max_, h_graph_sum_), 1)            
            # H_.append(h_graph_)

        h_k = torch.cat(H, dim=1)
        x = self.bn_out(h_k)
        # x = F.relu(self.lin1(x))
        x = F.leaky_relu(self.lin1(x))
        # x = self.dropout(x)
        # x = F.relu(self.lin2(x))
        x = F.leaky_relu(self.lin2(x))
        # x = self.dropout(x)
        x = self.out_fun(self.lin3(x))
        
        # h_k_ = torch.cat(H_, dim=1)
        # x_ = self.bn_out(h_k)
        # x_ = self.lin1_(x_)
        # x_ = self.dropout(x_)
        # x_ = self.lin2_(x_)
        # x_ = self.dropout(x_)
        # x_validity = self.out_fun(self.validity_pred(x_))
        x_validity = x
        
        return x, x_validity
