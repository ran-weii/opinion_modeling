import torch
import torch.nn as nn

from src.models.gnn import TransformerConv

class MLP(nn.Module):
    """ Multilayer perceptron backbone """
    def __init__(self, input_dim, hidden_dim, num_layers, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "softplus":
            self.act = nn.Softplus()
        else:
            raise NotImplementedError
        
        last_dim = input_dim
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return x


class GNN(nn.Module):
    """ Graph neural network backbone with Transformer and MLP layers """
    def __init__(self, input_dim, hidden_dim, num_gnn_layers, num_heads, num_mlp_layers, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads
        self.num_mlp_layers = num_mlp_layers
        self.activation = activation

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "softplus":
            self.act = nn.Softplus()
        else:
            raise NotImplementedError
        
        last_dim = input_dim
        out_dim = hidden_dim // num_heads
        gnn_layers = []
        for _ in range(num_gnn_layers):
            gnn_layers.append(TransformerConv(last_dim, out_dim, heads=num_heads))
            last_dim = out_dim * num_heads
            out_dim = hidden_dim // num_heads
        self.gnn_layers = nn.ModuleList(gnn_layers)

        mlp_layers = []
        for _ in range(num_mlp_layers):
            mlp_layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, x, adj):
        for layer in self.gnn_layers:
            x = self.act(layer(x, adj))

        for layer in self.mlp_layers:
            x = self.act(layer(x))
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    input_dim = 10
    hidden_dim = 24
    num_layers = 2
    activation = "relu"
    
    # test mlp backbone
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    mlp = MLP(
        input_dim, hidden_dim, num_layers, activation
    )
    print(mlp)
    out = mlp(x)

    assert list(out.shape) == [batch_size, hidden_dim]
    print("mlp backbone passed\n")

    # test gnn backbone
    num_gnn_layers = 2
    num_heads = 2
    num_mlp_layers = 2
    
    batch_size = 32
    num_nodes = 64
    x = torch.randn(batch_size, num_nodes, input_dim)
    adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).to(torch.float32)
    
    gnn = GNN(
        input_dim, hidden_dim, num_gnn_layers, num_heads, num_mlp_layers, activation
    )
    print(gnn)
    out = gnn(x, adj)

    assert list(out.shape) == [batch_size, num_nodes, hidden_dim]
    print("gnn backbone passed\n")