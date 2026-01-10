"""
Last modified date: 2023.07.31
Author: Jialiang Zhang
Description: pointnet
"""

import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(
        self, 
        point_feature_dim=3, 
        local_conv_hidden_layers_dim=[64, 128, 256], 
        global_mlp_hidden_layers_dim=[256], 
        pc_feature_dim=128, 
        activation = 'leaky_relu',
    ):
        super(PointNet, self).__init__()
        act_fn = dict(
            relu=nn.ReLU,
            leaky_relu=nn.LeakyReLU,
            elu=nn.ELU,
            tanh=nn.Tanh,
            gelu=nn.GELU,
        )[activation]
        self.point_feature_dim = point_feature_dim
        # build local 1d convolutional network
        self.local_conv = nn.Sequential()
        self.local_conv.add_module('conv0', nn.Conv1d(point_feature_dim, local_conv_hidden_layers_dim[0], (1,)))
        self.local_conv.add_module('act0', act_fn())
        for i in range(1, len(local_conv_hidden_layers_dim)):
            self.local_conv.add_module(f'conv{i}', nn.Conv1d(local_conv_hidden_layers_dim[i - 1], local_conv_hidden_layers_dim[i], (1,)))
            self.local_conv.add_module(f'act{i}', act_fn())
        # build global mlp
        if len(global_mlp_hidden_layers_dim):
            self.has_global_mlp = True
            self.global_mlp = nn.Sequential()
            self.global_mlp.add_module('linear0', nn.Linear(local_conv_hidden_layers_dim[-1], global_mlp_hidden_layers_dim[0]))
            self.global_mlp.add_module('act0', act_fn())
            for i in range(1, len(global_mlp_hidden_layers_dim)):
                self.global_mlp.add_module(f'linear{i}', nn.Linear(global_mlp_hidden_layers_dim[i - 1], global_mlp_hidden_layers_dim[i]))
                self.global_mlp.add_module(f'act{i}', act_fn())
            self.global_mlp.add_module(f'linear{len(global_mlp_hidden_layers_dim)}', nn.Linear(global_mlp_hidden_layers_dim[-1], pc_feature_dim))
        else:
            self.has_global_mlp = False
        # initialize weights
        for module in self.local_conv:
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_normal_(module.weight)
        if len(global_mlp_hidden_layers_dim):
            for module in self.global_mlp:
                if isinstance(module, nn.Conv1d):
                    nn.init.xavier_normal_(module.weight)

    def forward(self, x):
        # x: (batch_size, num_points, input_feature_dim)
        # global_feature: (batch_size, output_feature_dim)
        x = x[..., :self.point_feature_dim]
        local_feature = self.local_conv(x.transpose(1, 2))
        global_feature = local_feature.max(dim=2)[0]
        if self.has_global_mlp:
            global_feature = self.global_mlp(global_feature)
        return global_feature, local_feature.max(dim=2)[1] #, local_feature.max(dim=2)[0]
