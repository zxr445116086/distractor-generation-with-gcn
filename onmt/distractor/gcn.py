import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """
    A GCN/Contextualized GCN module operated on adjacency matrix.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.8):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gcn_drop = nn.Dropout(dropout)
    
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.input_dim if layer == 0 else self.hidden_dim
            self.W.append(nn.Linear(input_dim, self.hidden_dim))

    def forward(self, adj, gcn_inputs):
        """
        :param adj: batch_size * num_vertex * num_vertex
        :param gcn_inputs: batch_size * num_vertex * input_dim
        :return: gcn_outputs: list of batch_size * num_vertex * hidden_dim
                 mask: batch_size * num_vertex * 1. In mask, 1 denotes
                     this vertex is PAD vertex, 0 denotes true vertex.
        """
        # use out degree, assume undirected graph
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        gcn_outputs = []
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
            gcn_outputs.append(gcn_inputs)
        gcn_outputs = torch.cat(gcn_outputs, 2)

        return gcn_outputs, mask
