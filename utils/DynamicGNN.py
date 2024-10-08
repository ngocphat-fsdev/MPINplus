import copy
import os
import random
import sys
import time
sys.path.append('/home/xiao/Documents/OCW')
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

from torch_geometric.nn import knn_graph,radius_graph

# class DynamicEdgeConv(EdgeConv):
#     def __init__(self, in_channels, out_channels, k=6):
#         super().__init__(in_channels, out_channels)
#         self.k = k
#
#     def forward(self, x, batch=None):
#         edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
#         return super().forward(x, edge_index)


class DynamicGAT(pyg_nn.GATConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGATv2(pyg_nn.GATv2Conv):
    # need to upgrade the version of pytorch geometric to use this pyg_nn.GATv2Conv

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGCN(pyg_nn.GCNConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGCN, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGraphSAGE(pyg_nn.SAGEConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        # default is mean, by changing the aggr to max, it becomes max pooling
        super(DynamicGraphSAGE, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class GraphSAGEPlusPlusDA(pyg_nn.MessagePassing):
    def __init__(self, in_channels, hidden_channels_list, out_channels, k=None, radius=None):
        super(GraphSAGEPlusPlusDA, self).__init__(aggr='mean')  # or other aggregation if needed
        self.k = k
        self.radius = radius
        self.num_layers = len(hidden_channels_list)
        self.convs_mean = nn.ModuleList()
        self.convs_max = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels_layer = in_channels if i == 0 else 2 * hidden_channels_list[i-1]
            out_channels_layer = hidden_channels_list[i]
            self.convs_mean.append(pyg_nn.SAGEConv(in_channels_layer, out_channels_layer, aggr='mean'))
            self.convs_max.append(pyg_nn.SAGEConv(in_channels_layer, out_channels_layer, aggr='max'))

        self.post_mp = nn.Linear(2 * hidden_channels_list[-1], out_channels)

    def reset_parameters(self):
        for conv in self.convs_mean:
            conv.reset_parameters()
        for conv in self.convs_max:
            conv.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, batch=None):
        # Dynamically create edge_index using knn_graph or radius_graph
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        elif self.radius is not None:
            edge_index = radius_graph(x, self.radius, loop=False)
        else:
            raise ValueError("Either k or radius must be provided.")

        all_layers = []
        for i in range(self.num_layers):
            x_mean = self.convs_mean[i](x, edge_index)
            x_max = self.convs_max[i](x, edge_index)

            # Apply ReLU activation function
            x_mean = F.relu(x_mean)
            x_max = F.relu(x_max)

            # Concatenate the mean and max features
            x = torch.cat([x_mean, x_max], dim=1)
            all_layers.append(x)

        # Apply the post message-passing layer
        x_final = self.post_mp(x)
        return F.log_softmax(x_final, dim=-1)

class StaticGraphSAGE(pyg_nn.SAGEConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGraphSAGE, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index
 

class StaticGCN(pyg_nn.GCNConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGCN, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index

class StaticGAT(pyg_nn.GATConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index



