import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_features, n_nodes, bias=True, activation=None, dropout_rate=0):
        """
        in_features: Number of input features (int)
        n_nodes: Number of nodes in the layer (int)
        bias: Whether to include a bias term (bool)
        activation: Activation function to use (callable)
        dropout_rate: Dropout rate for nodes (float)
        """
        super(DenseLayer, self).__init__()
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(n_nodes, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_nodes))
        else:
            self.bias = None
        
        # Initialize weights and biases using xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        # Store activation and dropout
        self.activation = activation
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None
            
    def forward(self, x):
        """
        x: Input tensor (batch_size, in_features)
        """
        # Perform dense operation
        out = x.mm(self.weight.t())
        if self.bias is not None:
            out += self.bias
        
        # Apply activation and dropout
        if self.activation is not None:
            out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
