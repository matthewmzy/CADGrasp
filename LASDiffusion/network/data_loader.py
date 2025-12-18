import torch
from pathlib import Path
import numpy as np
import os, sys
from random import random, randint
from tqdm import tqdm
from ipdb import set_trace
import plotly.graph_objects as go
from termcolor import cprint
from time import time

from IBSGrasp.utils.transforms import batch_transform_points, transform_points

def pc_plotly(pc, size=3, color='green'):
    return go.Scatter3d(
                x=pc[:, 0] if isinstance(pc, np.ndarray) else pc[:, 0].numpy(),
                y=pc[:, 1] if isinstance(pc, np.ndarray) else pc[:, 1].numpy(),
                z=pc[:, 2] if isinstance(pc, np.ndarray) else pc[:, 2].numpy(),
                mode='markers',
                marker=dict(size=size, color=color),
            )

class IBS_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, scene_path, ibs_load_per_scene, split, overfit=False, scaling=1):
        super().__init__()
        assert split in ('train', 'test')
        self.path = path
        self.scene_path = scene_path
        self.split = split
        self.overfit = overfit
        self.ibs_load_per_scene=ibs_load_per_scene
        self.vis_num=0
        self.total_vis_num = 3
        self.scaling=scaling

        if split == 'train':
            scene_ids = [i for i in range(100)]
        else:
            scene_ids = [i for i in range(5)]
        self.items = []
        view_ids = range(256)
        for scene_id in scene_ids:
            for view_id in view_ids:
                vp = f'IBSGrasp/scene_valid_ids/scene_{str(scene_id).zfill(4)}/view_{str(view_id).zfill(4)}.npy'
                if not os.path.exists(vp):
                    continue
                valid_npy = np.load(vp)
                valid_num = np.sum(valid_npy)
                true_indices = np.where(valid_npy == True)[0]
                random_indices = np.random.choice(true_indices, size=valid_num-(valid_num//self.scaling), replace=False)
                valid_npy[random_indices]=False
                valid_num = np.sum(valid_npy)
                if valid_num >= self.ibs_load_per_scene:
                    self.items.append((scene_id, view_id, valid_npy))
        cprint(f"Loaded {len(self.items)} scene views")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        scene_id, view_id, valid_ids = self.items[idx]
        scene_name = 'scene_' + str(scene_id).zfill(4)
        ibs_path = os.path.join(self.path, 'ibs', scene_name + '.npy')
        w2h_trans_path = os.path.join(self.path, 'w2h_trans', scene_name + '.npy')
        perm = np.random.permutation(np.sum(valid_ids))[:self.ibs_load_per_scene]
        ibs = np.load(ibs_path)[valid_ids][perm]
        w2h_trans = np.load(w2h_trans_path)[valid_ids][perm]
        scene_data = np.load(os.path.join(self.scene_path, 'scene_' + str(scene_id).zfill(4), 'realsense', 'network_input.npz'))
        scene_pc_c = scene_data['pc'][view_id]  # (N, 3)
        scene_pc_c = torch.from_numpy(scene_pc_c).float()  # 保持在CPU
        c2w_trans = scene_data['extrinsics'][view_id]
        c2w_trans = torch.from_numpy(c2w_trans).float()  # 保持在CPU
        scene_pc_w = transform_points(scene_pc_c, c2w_trans)  # (N, 3)，世界坐标系
        scene_pc_h = batch_transform_points(scene_pc_w.unsqueeze(0).repeat(self.ibs_load_per_scene, 1, 1), torch.from_numpy(w2h_trans).float())

        # IBS体素化 (Batched)
        B = self.ibs_load_per_scene
        ibs = torch.from_numpy(ibs).to(dtype=torch.bool)  # 形状: (B, D, H, W, C)
        ibs_voxel_float = torch.full((B, *ibs.shape[1:4], 2), -1.0, dtype=torch.float)  # 形状: (B, D, H, W, 2)
        ibs_voxel_float[..., 0] = torch.where(ibs[..., 0], 1.0, -1.0)  # 占用
        ibs_voxel_float[..., 1] = torch.where(ibs[..., 1], 1.0, ibs_voxel_float[..., 1])  # 其他手指接触
        ibs_voxel_float[..., 1] = torch.where(ibs[..., 2], 2.0, ibs_voxel_float[..., 1])  # 大拇指接触
        ibs_voxel_float = ibs_voxel_float.permute(0, 4, 1, 2, 3)  # 形状: (B, 2, D, H, W)

        # 场景点云体素化 (Batched)
        x_min, x_max = -0.1, 0.1
        y_min, y_max = -0.1, 0.1
        z_min, z_max = -0.1, 0.1
        resolution = 0.005
        grid_size = int((x_max - x_min) / resolution)
        scene_voxel_float = torch.zeros((B, grid_size, grid_size, grid_size), dtype=torch.float)
        min_coords = torch.tensor([x_min, y_min, z_min])
        for i in range(B):
            scene_pc_h_i = scene_pc_h[i]
            mask = (scene_pc_h_i[:, 0] > x_min) & (scene_pc_h_i[:, 0] < x_max) & \
                (scene_pc_h_i[:, 1] > y_min) & (scene_pc_h_i[:, 1] < y_max) & \
                (scene_pc_h_i[:, 2] > z_min) & (scene_pc_h_i[:, 2] < z_max)
            scene_pc_h_i = scene_pc_h_i[mask]
            scene_pc_h_i = (scene_pc_h_i - torch.tensor([x_min, y_min, z_min], device=scene_pc_h_i.device)) // resolution
            scene_pc_h_i = scene_pc_h_i.to(torch.int)
            scene_voxel_float[i, scene_pc_h_i[:, 0], scene_pc_h_i[:, 1], scene_pc_h_i[:, 2]] = 1
        scene_voxel_float = scene_voxel_float.unsqueeze(1)  # 形状: (B, 1, grid_size, grid_size, grid_size)

        return {"ibs": ibs_voxel_float, "scene": scene_voxel_float}
