import torch
import torch.nn as nn
from nflows.nn.nets.resnet import ResidualNet

class ConditionalTransform(nn.Module):
    '''
    A mlp with 2 residual blocks
    '''
    def __init__(self, Ni, No, Nh=64, mask=None):
        super(ConditionalTransform, self).__init__()
        self.mask = mask
        self.net = ResidualNet(Ni, No, hidden_features=Nh, num_blocks=2, dropout_probability=0.0, use_batch_norm=False)

    def forward(self, x: torch.Tensor):
        return self.net(x)
