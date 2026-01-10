"""
Visualize DexGraspNet 2.0 Grasp Annotations.

Visualize grasp data from data/DexGraspNet2.0/dex_grasps_new.
Supports visualizing raw grasps, success-filtered grasps, or FPS-sampled grasps.

Usage:
    python tests/visualize_dex_grasp.py --scene scene_0055 --view 0000
    python tests/visualize_dex_grasp.py --scene scene_0055 --use_success --grasp_num 5
    python tests/visualize_dex_grasp.py --scene scene_0055 --use_fps --grasp_num 10
"""

import os
import sys

import argparse
import random
import numpy as np
import torch
import plotly.express as px

from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.baseline.utils.util import set_seed

from cadgrasp.paths import project_path

# Default paths (DexGraspNet2.0 structure)
DEFAULT_GRASP_PATH = project_path('data/DexGraspNet2.0/dex_grasps_new')
DEFAULT_SUCCESS_PATH = project_path('data/DexGraspNet2.0/dex_grasps_success_indices')
DEFAULT_FPS_PATH = project_path('data/DexGraspNet2.0/fps_sampled_indices')
DEFAULT_SCENE_PATH = project_path('data/DexGraspNet2.0/scenes')


def load_grasps(scene: str, robot_name: str, grasp_path: str, 
                success_path: str = None, fps_path: str = None,
                use_success: bool = False, use_fps: bool = False):
    """
    Load grasp data with optional filtering.
    
    Args:
        scene: Scene name (e.g., 'scene_0055')
        robot_name: Robot name (e.g., 'leap_hand')
        grasp_path: Path to grasp data
        success_path: Path to success indices
        fps_path: Path to FPS indices
        use_success: Filter by success indices
        use_fps: Filter by FPS indices (implies use_success)
    
    Returns:
        Dict mapping object_id to grasp data dict
    """
    scene_grasp_path = os.path.join(grasp_path, scene, robot_name)
    
    if not os.path.exists(scene_grasp_path):
        print(f"Grasp path not found: {scene_grasp_path}")
        return {}
    
    result = {}
    for npz_file in sorted(os.listdir(scene_grasp_path)):
        if not npz_file.endswith('.npz'):
            continue
        
        # Load grasp data
        data = np.load(os.path.join(scene_grasp_path, npz_file))
        data = {k: data[k] for k in data.files}
        n_grasps = len(data['translation'])
        
        # Apply success filter
        if use_success or use_fps:
            success_file = os.path.join(success_path, scene, robot_name, npz_file)
            if os.path.exists(success_file):
                success_data = np.load(success_file)
                success_indices = success_data['success_indices']
                data = {k: v[success_indices] for k, v in data.items()}
            else:
                # No success file, assume all successful
                pass
        
        # Apply FPS filter
        if use_fps:
            fps_file = os.path.join(fps_path, scene, robot_name, npz_file)
            if os.path.exists(fps_file):
                fps_data = np.load(fps_file)
                fps_indices = fps_data['fps_indices']
                data = {k: v[fps_indices] for k, v in data.items()}
        
        if len(data['translation']) > 0:
            result[npz_file] = data
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Visualize DexGraspNet grasp annotations")
    parser.add_argument('--robot_name', type=str, default='leap_hand', choices=['leap_hand'])
    parser.add_argument('--urdf_path', type=str, default='robot_models/urdf/leap_hand_simplified.urdf')
    parser.add_argument('--meta_path', type=str, default='robot_models/meta/leap_hand/meta.yaml')
    parser.add_argument('--camera', type=str, default='realsense')
    parser.add_argument('--scene', type=str, default='scene_0055')
    parser.add_argument('--view', type=str, default='0000')
    parser.add_argument('--grasp_num', type=int, default=3, help='Number of grasps to visualize per object')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--frame', type=str, default='world', choices=['world', 'camera'])
    parser.add_argument('--use_success', action='store_true', help='Use success-filtered grasps')
    parser.add_argument('--use_fps', action='store_true', help='Use FPS-sampled grasps (implies success)')
    parser.add_argument('--grasp_path', type=str, default=DEFAULT_GRASP_PATH)
    parser.add_argument('--scene_path', type=str, default=DEFAULT_SCENE_PATH)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    vis = Vis(
        robot_name=args.robot_name,
        urdf_path=args.urdf_path,
        meta_path=args.meta_path,
    )

    # Load scene point cloud
    view_plotly, pc, extrinsics = vis.scene_plotly(
        args.scene, args.view, args.camera, 
        with_pc=True, mode='pc', with_extrinsics=True
    )
    
    # Camera transforms
    cam0_wrt_table = np.load(os.path.join(args.scene_path, args.scene, args.camera, 'cam0_wrt_table.npy'))
    camera_poses = np.load(os.path.join(args.scene_path, args.scene, args.camera, 'camera_poses.npy'))
    camera_pose_wrt_cam0 = camera_poses[int(args.view)]
    camera_pose = torch.from_numpy(np.einsum('ab,bc->ac', cam0_wrt_table, camera_pose_wrt_cam0)).float()
    
    # Transform PC to world frame if needed
    if args.frame == 'world':
        pc = pc.float()
        pc[:, :3] = torch.einsum('ab,nb->na', camera_pose[:3, :3], pc[:, :3]) + camera_pose[:3, 3]
    
    # Subsample PC for visualization
    idxs = torch.randperm(len(pc))[:10000]
    view_plotly = vis.pc_plotly(pc[idxs, :3], size=1, color='lightblue')
    
    # Load grasps
    grasps = load_grasps(
        args.scene, args.robot_name, args.grasp_path,
        DEFAULT_SUCCESS_PATH, DEFAULT_FPS_PATH,
        args.use_success, args.use_fps
    )
    
    filter_type = "FPS" if args.use_fps else ("Success" if args.use_success else "Raw")
    total_grasps = sum(len(g['translation']) for g in grasps.values())
    print(f"Scene {args.scene}: {len(grasps)} objects, {total_grasps} {filter_type} grasps")
    
    robot_plotly = []
    point_plotly = []
    
    for obj_id, data in grasps.items():
        n_grasps = len(data['translation'])
        idxs = np.random.choice(n_grasps, min(args.grasp_num, n_grasps), replace=False)
        
        for i in idxs:
            # Transform grasp to correct frame
            trans = torch.from_numpy(
                np.einsum('ba,b->a', camera_pose_wrt_cam0[:3, :3], 
                         data['translation'][i] - camera_pose_wrt_cam0[:3, 3])
            )
            rot = torch.from_numpy(
                np.einsum('ba,bc->ac', camera_pose_wrt_cam0[:3, :3], data['rotation'][i])
            )
            point = torch.from_numpy(
                np.einsum('ba,b->a', camera_pose_wrt_cam0[:3, :3],
                         data['point'][i] - camera_pose_wrt_cam0[:3, 3])
            )
            
            if args.frame == 'world':
                trans = torch.einsum('ab,b->a', camera_pose[:3, :3], trans.float()) + camera_pose[:3, 3]
                rot = torch.einsum('ab,bc->ac', camera_pose[:3, :3], rot.float())
                point = torch.einsum('ab,b->a', camera_pose[:3, :3], point.float()) + camera_pose[:3, 3]
            
            qpos = {k: torch.from_numpy(data[k][[i]]).float() for k in data.keys()}
            
            robot_plotly += vis.robot_plotly(
                trans[None].float(), rot[None].float(), qpos, 
                opacity=0.6, color=random.choice(px.colors.qualitative.Set1)
            )
            point_plotly += vis.pc_plotly(point[None].float(), size=8, color='red')
    
    plotly = view_plotly + robot_plotly + point_plotly
    vis.show(plotly, args.output_path)


if __name__ == '__main__':
    main()