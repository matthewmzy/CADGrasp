"""
Last modified date: 2023.08.01
Author: Jialiang Zhang
Description: mlp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_layers_dim, 
        output_dim, 
        act=None,
        use_layer_norm=False,
    ):
        super().__init__()
        act = 'leaky_relu' if act is None else act
        act_fn = dict(
            relu=nn.ReLU,
            leaky_relu=nn.LeakyReLU,
            mish=Mish, 
            elu=nn.ELU,
            tanh=nn.Tanh,
        )[act]
        # build mlp
        hidden_layers_dim = deepcopy(hidden_layers_dim)
        hidden_layers_dim.insert(0, input_dim)
        self.mlp = nn.Sequential()
        for i in range(1, len(hidden_layers_dim)):
            self.mlp.add_module(f'linear{i - 1}', nn.Linear(hidden_layers_dim[i - 1], hidden_layers_dim[i]))
            # self.mlp.add_module(f'bn{i - 1}', nn.BatchNorm1d(hidden_layers_dim[i]))
            if use_layer_norm:
                self.mlp.add_module(f'ln{i - 1}', nn.LayerNorm(hidden_layers_dim[i]))
            self.mlp.add_module(f'act{i - 1}', act_fn())
        self.mlp.add_module(f'linear{len(hidden_layers_dim) - 1}', nn.Linear(hidden_layers_dim[-1], output_dim))
        # initialize weights
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
    
    def forward(self, x):
        return self.mlp(x)