import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .conv import ResNet14D
from .conv_unet import MinkUNet14D


def get_backbone(
    backbone_name: str, 
    feature_dim: int, 
    backbone_config: dict
):
    """
    Get backbone model from config
    
    Args: 
    - backbone: str, backbone name
    - feature_dim: int, feature dimension
    - backbone_config: dict, backbone config
    
    Returns:
    - backbone: nn.Module, backbone model
    """
    if backbone_name == "sparseconv":
        return MinkUNet14D(
            in_channels=3, out_channels=feature_dim, D=3)
    elif backbone_name == "sparse_glob_conv":
        return ResNet14D(
            in_channels=3, out_channels=feature_dim, D=3)
    else:
        raise ValueError(f"Backbone {backbone_name} not supported")

def get_feature(
    backbone_name: str, 
    backbone: nn.Module, 
    data: dict
):
    """
    Get feature from backbone
    
    Args:
    - backbone_name: str, backbone name
    - backbone: nn.Module, backbone model
    - data: dict, input data, format: {
        'point_clouds': torch.Tensor[B, N, 3, torch.float32], point clouds
        'coors': torch.Tensor[M, 4, torch.int32], batch id and coordinates
        'feats': torch.Tensor[M, 3, torch.float32], features, ones
        'original2quantize': torch.Tensor[M, torch.int64], original2quantize
        'quantize2original': torch.Tensor[B * N, torch.int64], quantize2original
    }
    
    Returns:
    - feature: torch.Tensor[B, N, C], feature
    """
    pc = data['point_clouds']
    batch_size, point_num, _ = pc.shape
    if backbone_name in ['sparseconv', 'sparse_glob_conv']:
        coor = data['coors']
        feat = data['feats']
        mink_input = ME.SparseTensor(feat, coordinates=coor)
        mink_output = backbone(mink_input).F
        if backbone_name == 'sparseconv':
            feature = mink_output[data['quantize2original']].view(batch_size, point_num, -1)
        else:
            feature = mink_output
    else:
        raise ValueError(f"Backbone {backbone_name} not supported")
    return feature
