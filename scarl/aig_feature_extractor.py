# BSD 3-Clause License

# Copyright (c) 2024, NYU Machine-Learning guided Design Automation (MLDA)
# Author: Animesh Basak Chowdhury
# Date: March 9, 2024

import torch
import torch.nn as nn
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
    def __init__(self,node_encoder_out_channels=4,hidden_channels=16):
        super().__init__()

        #self.node_encoder = NodeEncoder()
        #self.sage_aggr = MultiAggregation([SoftmaxAggregation(t=0.1,learn=True)])
        #self.conv1 = SAGEConv(node_encoder_out_channels,hidden_channels,aggr=self.sage_aggr)
        #self.conv2 = SAGEConv(hidden_channels,hidden_channels,aggr=self.sage_aggr)
        #self.norm1 = BatchNorm(hidden_channels)
        #self.norm2 = BatchNorm(hidden_channels)
        # Use a global sort aggregation:
        self.synthesis_transform_feature_dim=1
        self.num_layers = 2
        self.hidden_size = 64
        #self.global_pool_1 = aggr.MLPAggregation(hidden_channels,hidden_channels,max_num_elements=500,num_layers=1)
        #self.global_pool_2 = aggr.GRUAggregation(hidden_channels,hidden_channels)
        self.lstm = nn.LSTM(self.synthesis_transform_feature_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        #self.global_pool_3 = aggr.SetTransformerAggregation(hidden_channels,num_seed_points=1,heads=1)
        #self.linear_layer = torch.nn.Linear(hidden_channels*2, hidden_channels*2)
        #self.linear_layer = torch.nn.Linear(self.hidden_size,self.hidden_size)
        #self.features_dim = self.hidden_size+(hidden_channels*2)
        self.features_dim = self.hidden_size
        self.linear_layer = torch.nn.Linear(self.features_dim,self.features_dim)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight.data)
        self.init_weights()
        
        
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)

    def forward(self,batched_data):
        # x = batched_data.x
        # edge_index = batched_data.edge_index
        # batch = batched_data.batch
        # x = self.node_encoder(x)
        # x = F.relu(self.norm1(self.conv1(x, edge_index)))
        # x = F.relu(self.norm2(self.conv2(x, edge_index)))
        #pooled_feature = torch.cat([self.global_pool_1(x, batch),self.global_pool_2(x,batch),self.global_pool_3(x,batch)],dim=1)
        # pooled_feature = torch.cat([self.global_pool_1(x, batch),self.global_pool_2(x,batch)],dim=1)
        # graph_feature = self.linear_layer(pooled_feature)
        # return graph_feature
        recipe = batched_data.recipe_encoding
        # #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # # Initialize cell state
        # #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # # Forward propagate LSTM
        # #out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(recipe, None)
        
        # Decode the hidden state of the last time step
        out = self.linear_layer(out[:, -1, :])
        return out
        # pooled_feature = torch.cat([self.global_pool_1(x, batch),self.global_pool_2(x,batch),out[:, -1, :]],dim=1)
        # graph_feature = self.linear_layer(pooled_feature)
        # return graph_feature
        
class RLValueNet(torch.nn.Module):
    def __init__(self,hidden_channels=16):
        super().__init__()
        self.linear_layer_1 = torch.nn.Linear(hidden_channels,hidden_channels)
        self.linear_layer_2 = torch.nn.Linear(hidden_channels,1)
        torch.nn.init.xavier_uniform_(self.linear_layer_1.weight.data)
        torch.nn.init.xavier_uniform_(self.linear_layer_2.weight.data)


    def forward(self,x):
        x = F.relu(self.linear_layer_1(x))
        x = F.tanh(self.linear_layer_2(x))
        return x