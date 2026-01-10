import os
import sys

import argparse
from tqdm import trange
import numpy as np
import torch
import scipy.io as scio
from pytorch3d import transforms as pttf
from pytorch3d.ops import knn_points

from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.baseline.utils.pose_refine import PoseRefine

camera = 'realsense'

def main():
    refiner = PoseRefine()
    for i in range(100):
        scene = f'scene_{str(i).zfill(4)}'
        poses = torch.from_numpy(np.load(os.path.join('data', 'poses_gn', scene, camera, 'poses.npy')))

        camera_poses = torch.from_numpy(np.load(os.path.join('data', 'scenes', scene, camera, 'camera_poses.npy'))[0])
        align_mat = torch.from_numpy(np.load(os.path.join('data', 'scenes', scene, camera, 'cam0_wrt_table.npy')))

        view = '0000'
        meta = scio.loadmat(os.path.join('data', 'scenes', scene, camera, 'meta', view + '.mat'))
        obj_idxs = list(meta['cls_indexes'].flatten().astype(np.int32))
        obj_poses = torch.from_numpy(meta['poses'])
        extra = [(idx-1, obj_poses[:, :, i]) for i, idx in enumerate(obj_idxs)]

        rot = poses[:, -13:-4].reshape(-1, 3, 3)
        trans = poses[:, -4:-1]
        new_rot = torch.einsum('ji,njk->nik', camera_poses[:3, :3], rot)
        new_trans = torch.einsum('ji,nj->ni', camera_poses[:3, :3], trans - camera_poses[:3, 3])

        idxs = []
        grasp_points = []

        for i in trange(len(poses)):
            try:
                table_mat = torch.linalg.inv(align_mat @ camera_poses)
                table_vec = torch.cat([table_mat[:3, 2], -(table_mat[:3, 2] * table_mat[:3, 3]).sum(0, keepdim=True)])
                new_rot[i], new_trans[i], poses[i, 1], poses[i, 3], grasp_point = refiner.refine(new_rot[i], new_trans[i], poses[i, 1], poses[i, 3], poses[i, -1].int().item(), obj_poses[:, :, obj_idxs.index(poses[i, -1]+1)], table=table_vec, extra=extra)

                idxs.append(i)
                grasp_points.append(grasp_point)
            except:
                pass

        print(f'{len(idxs)} in {len(poses)}: {len(idxs)/len(poses)}')
        new_rot = torch.einsum('ij,njk->nik', camera_poses[:3, :3], new_rot)
        new_trans = torch.einsum('ij,nj->ni', camera_poses[:3, :3], new_trans) + camera_poses[:3, 3]
        poses = torch.cat([poses[:, :-13], new_rot.reshape(-1, 9), new_trans, poses[:, -1:]], dim=-1)[idxs]
        path = os.path.join('data', 'gripper_grasps', scene, camera)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'poses.npy'), poses.numpy())
        np.save(os.path.join(path, 'points.npy'), torch.stack(grasp_points).numpy())

main()