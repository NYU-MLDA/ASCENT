# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import MessagePassing, aggr, SAGEConv,MultiAggregation,SoftmaxAggregation,BatchNorm
import torch.nn.functional as F
import torch as th

allowable_features = {
    'node_type' : [0,1,2],
    'num_inverted_predecessors' : [0,1,2]
}

def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))

full_node_feature_dims = get_node_feature_dims()

class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim=3):
        super(NodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        x_embedding = self.node_type_embedding(x[:, 0])
        x_embedding = torch.cat((x_embedding, x[:,1].reshape(-1,1)), dim=1)
        #x_embedding = self.node_type_embedding(x)
        return x_embedding

class AIGStateEncoder(torch.nn.Module):
    def __init__(self,node_encoder_out_channels,hidden_channels):
        super().__init__()

        self.node_encoder = NodeEncoder()
        self.sage_aggr = MultiAggregation([SoftmaxAggregation(t=0.1,learn=True)])
        self.conv1 = SAGEConv(node_encoder_out_channels,hidden_channels,aggr=self.sage_aggr)
        self.conv2 = SAGEConv(hidden_channels,hidden_channels,aggr=self.sage_aggr)
        self.norm1 = BatchNorm(hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        # Use a global sort aggregation:
        self.global_pool_1 = aggr.MLPAggregation(hidden_channels,hidden_channels)
        self.global_pool_2 = aggr.GRUAggregation(hidden_channels,hidden_channels)
        self.global_pool_3 = aggr.SetTransformerAggregation(hidden_channels,hidden_channels)
        self.linear_layer = torch.nn.Linear(hidden_channels*3, hidden_channels*3)
        self.features_dim = hidden_channels*3

    def foward(self, x, edge_index, batch):
        x = self.node_encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        pooled_feature = torch.cat([self.global_pool(x, batch),self.global_pool_2(x,batch),self.global_pool_3(x,batch)],dim=1)
        graph_feature = self.classifier(pooled_feature)
        return graph_feature