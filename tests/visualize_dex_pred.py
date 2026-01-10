"""
Visualize DexGrasp Network Predictions.

Load a trained dexterous grasping network checkpoint and visualize predicted grasps.
This script requires a trained baseline model checkpoint.

Note: This script depends on the baseline network code and data paths.
Ensure data/scenes is properly linked or configured.

Usage:
    python tests/visualize_dex_pred.py --ckpt_path path/to/ckpt.pth --scene scene_0055
"""

import os
import sys

import argparse
import numpy as np
import torch
import random
import plotly.express as px
from pytorch3d import transforms as pttf
from pytorch3d.ops import sample_farthest_points

from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.baseline.utils.config import ckpt_to_config
from cadgrasp.baseline.network.model import get_model
from cadgrasp.baseline.utils.dataset import get_sparse_tensor
from cadgrasp.baseline.utils.util import pack_17dgrasp, unpack_17dgrasp, set_seed
from cadgrasp.baseline.utils.robot_info import GRIPPER_NEW_DEPTH

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--robot_name', type=str, default='leap_hand', choices=['leap_hand'])
parser.add_argument('--urdf_path', type=str, default='robot_models/urdf/leap_hand_simplified.urdf')
parser.add_argument('--meta_path', type=str, default='robot_models/meta/leap_hand/meta.yaml')
parser.add_argument('--camera', type=str, default='realsense')
parser.add_argument('--scene', type=str, default='scene_0090')
parser.add_argument('--view', type=str, default='0000')
parser.add_argument('--grasp_num', type=int, default=1024)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ratio', type=float, default=0.05)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cate', type=int, default=0)
parser.add_argument('--frame', type=str, default='world', choices=['world', 'camera'])
args = parser.parse_args()

set_seed(args.seed)

if __name__ == '__main__':
    vis = Vis(
        robot_name=args.robot_name,
        urdf_path=args.urdf_path,
        meta_path=args.meta_path,
    )

    _, pc = vis.scene_plotly(args.scene, args.view, args.camera, with_pc=True, mode='pc')
    seg = pc[:, 3]
    edge = pc[:, -1]
    pc = pc[:, :3]

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    config = ckpt_to_config(args.ckpt_path)
    model = get_model(config.model)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    model.config.voxel_size = config.data.voxel_size
    with torch.no_grad():
        data = get_sparse_tensor(pc[None].float(), config.data.voxel_size)
        data['seg'] = seg[None].long()
        edge = edge[None].to(device)
        data = {k: v.to(device) for k, v in data.items()}
        rot, trans, joints, score, obj_indices, seed_points = (t.cpu() for t in model.sample(data, args.grasp_num, edge=edge, graspness_scale=5, allow_fail=True, with_point=True, cate=False))

    sel_rots = []
    sel_trans = []
    sel_joints = []
    sel_points = []

    if args.cate:
        for obj_idx in obj_indices.unique():
            best_idx = torch.where(obj_indices[0] == obj_idx, score, 0*score-10000).argmax()
            sel_rots.append(rot[0, best_idx])
            sel_trans.append(trans[0, best_idx])
            sel_joints.append(joints[0, best_idx])
            sel_points.append(seed_points[best_idx])
    else:
        # best_idx = score.argmax()
        idxs = score.reshape(-1).sort(descending=True).indices
        for i in range(1):
            sel_rots.append(rot[0, idxs[i]])
            sel_trans.append(trans[0, idxs[i]])
            sel_joints.append(joints[0, idxs[i]])
            sel_points.append(seed_points[idxs[i]])

    sel_rots = torch.stack(sel_rots)
    sel_trans = torch.stack(sel_trans)
    sel_joints = torch.stack(sel_joints)
    sel_points = torch.stack(sel_points)
    
    pose_plotly = []
    kp_plotly = []
    for i in range(len(sel_rots)):
        pose_plotly += vis.robot_plotly(sel_trans[[i]].cpu(), sel_rots[[i]].cpu(), sel_joints[[i]].cpu(), opacity=1.0, color=random.choice(px.colors.sequential.Plasma))
        kp_plotly += vis.pc_plotly(sel_points[[i]].cpu(), size=5, color='red')

    pc_plotly = vis.pc_plotly(pc, color='blue')
    view_plotly, _ = vis.scene_plotly(args.scene, args.view, args.camera, mode='model', opacity=1.0)
    # plotly = view_plotly + pose_plotly + kp_plotly
    plotly = pc_plotly + pose_plotly + kp_plotly + view_plotly
    vis.show(plotly, args.output_path, scene=args.scene, view=args.view, camera=args.camera)
