import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConv(nn.Module):
    """ Graph convolution layer with multi-head attention """
    def __init__(
        self, 
        in_features, 
        out_features, 
        edge_features=1, 
        heads=1, 
        concat=True,
        dropout=0, 
        bias=True
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.bias = bias
        
        # attention weights
        self.lin_key = nn.Linear(in_features, heads * out_features)
        self.lin_query = nn.Linear(in_features, heads * out_features)
        self.lin_value = nn.Linear(in_features,  heads * out_features)
        
        # graph weights
        self.lin_edge = nn.Linear(edge_features, heads * out_features, bias=False)
        if concat:
            self.lin_root = nn.Linear(in_features, heads * out_features)
        else:
            self.lin_root = nn.Linear(in_features, out_features)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, adj):
        """ Assume same graph size
        args:
            x: [batch_size, num_nodes, node_features]
            adj: [batch_size, num_nodes, num_nodes, edge_features]

        returns:
            out: [batch_size, num_nodes, heads * out_features]
        """
        if adj.shape[1] != adj.shape[2]:
            raise ValueError("Adjacency matrix is not square matrix")
        if len(adj.shape) == 3:
            adj = adj.unsqueeze(-1)
        batch_size, num_nodes = x.shape[0], x.shape[1]

        # get neighbour message
        out = self.message(x, adj)
        if self.concat:
            out = out.view(batch_size, num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=2)
        
        # add root node
        out += self.lin_root(x)
        return out

    def message(self, x, adj):
        """ Calculate message from neighbours
        
        args:
            x: [batch_size, num_nodes, node_features]
            adj: [batch_size, num_nodes, num_nodes, edge_features]

        returns:
            out: [batch_size, node_dim, heads, out_features]
        """
        batch_size, num_nodes = x.shape[0], x.shape[1]
        A = 1 * (adj.sum(-1).abs() > 0).unsqueeze(-1)

        query = self.lin_query(x).view(batch_size, num_nodes, self.heads, self.out_features)
        key = self.lin_key(x).view(batch_size, num_nodes, self.heads, self.out_features)
        edge_attr = self.lin_edge(adj).view(batch_size, num_nodes, num_nodes, self.heads, self.out_features)

        # add edge features
        key = A.unsqueeze(-1) * key.unsqueeze(2) + edge_attr
        
        # compute attention
        alpha = torch.sum(query.unsqueeze(2) * key, dim=-1) / math.sqrt(self.out_features)
        mask = -1e10 * (1 - A) # mask non neighbours with -inf
        alpha = torch.softmax(alpha + mask, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # apply attention
        value = self.lin_value(x).view(batch_size, num_nodes, self.heads, self.out_features)
        out = A.unsqueeze(-1) * value.unsqueeze(1) + edge_attr
        out *= alpha.unsqueeze(-1)
        return out.sum(2)

    def __repr__(self):
        return "{}(in_features={}, out_features={}, edge_features={}, heads={}, dropout={}, bias={})".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.edge_features,
            self.heads,
            self.dropout,
            self.bias
        )
