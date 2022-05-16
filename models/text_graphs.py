import torch
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv, global_mean_pool
from torch.functional import F
from sklearn.metrics import average_precision_score, roc_auc_score



class ConversationalGraph(nn.Module):
    def __init__(self, embedding_size, hidden_channels, num_classes):
        channels= 128
        super(ConversationalGraph, self).__init__()
        self.gconv1 = GraphConv(embedding_size, hidden_channels)
        self.gconv2 = GraphConv(hidden_channels, channels)
        self.linear = nn.Linear(channels, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index, weights, batch):
        x = self.gconv1(x_embeddings, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        #Mean pooling
        x = global_mean_pool(x, batch)
        #Graph classification
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.linear(x)
        return out


class GATGraph(nn.Module):
    def __init__(self, embedding_size, hidden_channels, num_classes):
        channels= 128
        super(GATGraph, self).__init__()
        self.gconv1 = GATv2Conv(embedding_size, hidden_channels, heads=4, dropout=0.4)
        self.gconv2 = GraphConv(hidden_channels, channels, heads=1, concat=False, dropout=0.4)
        self.linear = nn.Linear(channels, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index, weights, batch):
        x = self.gconv1(x_embeddings, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        #Mean pooling
        x = global_mean_pool(x, batch)
        #Graph classification
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.linear(x)
        return out


class NodePrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.gconv1 = GCNConv(128, 128)
        self.gconv2 = GCNConv(128, 4)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index, weights):
        x = self.gconv1(x_embeddings, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index, weights)
        out = F.softmax(x, dim=1)
        return out
