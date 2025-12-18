import random
from typing import Optional
import numpy as np
import torch

from src.utils.robot_info import GRIPPER_HEIGHT

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def to_voxel_center(pc: torch.Tensor, voxel_size: float):
    """calculate the center of voxel corresponding to each point

    Args:
        pc (torch.Tensor): (..., 3)
    returns:
        voxel_center (torch.Tensor): (..., 3)
    """
    return torch.div(pc, voxel_size, rounding_mode='floor') * voxel_size + voxel_size / 2

def proper_svd(rot: torch.Tensor):
    """
    compute proper svd of rotation matrix
    rot: (B, 3, 3)
    return rotation matrix (B, 3, 3) with det = 1
    """
    u, s, v = torch.svd(rot.double())
    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum('bij,bkj->bik', u, v)))
        diag = torch.stack([torch.ones_like(s[:, 0]), torch.ones_like(s[:, 1]), sign], dim=-1)
        diag = torch.diag_embed(diag)
    return torch.einsum('bij,bjk,blk->bil', u, diag, v).to(rot.dtype)

def pack_17dgrasp(rot: torch.Tensor, # (N, 3, 3)
                trans: torch.Tensor, # (N, 3)
                width: torch.Tensor, # (N,)
                depth: torch.Tensor, # (N,)
                score: Optional[torch.Tensor] = None # (N,)
                ) -> np.ndarray:
    if score is None:
        score = torch.zeros_like(width)
    return torch.cat([score[:, None], width[:, None], torch.full_like(width[:, None], GRIPPER_HEIGHT), depth[:, None], rot.reshape(-1, 9), trans, torch.full_like(width[:, None], -1)], dim=-1).cpu().numpy()

def unpack_17dgrasp(grasp: np.ndarray):
    grasp = torch.from_numpy(grasp)
    return grasp[:, -13:-4].reshape(-1, 3, 3), grasp[:, -4:-1], grasp[:, 1], grasp[:, 3], grasp[:, 0]
