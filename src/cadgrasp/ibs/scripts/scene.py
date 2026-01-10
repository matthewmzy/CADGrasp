"""
Scene class for loading and managing scene data.

This module provides the Scene class for loading scene meshes, point clouds,
grasp data, and other scene-related information.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import random
import numpy as np
import torch
import transforms3d
import trimesh as tm
from termcolor import cprint
from tqdm import tqdm

from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.baseline.utils.util import set_seed
from copy import deepcopy
import plotly.express as px
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import pytorch3d


@dataclass
class SceneConfig:
    """Configuration for Scene class."""
    scene_id: int = 0
    robot_name: str = 'leap_hand'
    urdf_path: str = 'robot_models/urdf/leap_hand_simplified.urdf'
    meta_path: str = 'robot_models/meta/leap_hand/meta.yaml'
    camera: str = 'realsense'
    table_size: List[float] = field(default_factory=lambda: [0.6, 0.6, 0.0])
    device: str = 'cuda:0'
    visualize: bool = False
    grasp_num: int = 0  # 0 means don't load grasps
    num_samples: int = 4096
    output_path: str = ''
    with_graspness: bool = False
    load_gt_ibs: bool = False
    scene_base_path: str = 'data/DexGraspNet2.0/scenes'
    mesh_base_path: str = 'data/DexGraspNet2.0/meshdata'


class Scene:
    """
    Scene class for loading and managing scene data.
    
    This class handles:
    - Loading scene meshes and computing surface point clouds
    - Camera pose management
    - Grasp data loading and transformation
    - IBS data loading (optional)
    - Visualization
    
    Args:
        scene_id: Scene identifier (int or str like "scene_0055")
        robot_name: Robot name (default: 'leap_hand')
        urdf_path: Path to robot URDF
        meta_path: Path to robot meta file
        camera: Camera name (default: 'realsense')
        table_size: Table dimensions [x, y, z] (default: [0.6, 0.6, 0.0])
        device: Torch device (default: 'cuda:0')
        visualize: Whether to enable visualization (default: False)
        grasp_num: Number of grasps to load per object, 0 to skip (default: 0)
        num_samples: Number of surface points to sample (default: 4096)
        scene_base_path: Base path for scene data
        mesh_base_path: Base path for mesh data
        **kwargs: Additional configuration options
    """
    
    def __init__(
        self,
        scene_id: Union[int, str] = 0,
        robot_name: str = 'leap_hand',
        urdf_path: str = 'robot_models/urdf/leap_hand_simplified.urdf',
        meta_path: str = 'robot_models/meta/leap_hand/meta.yaml',
        camera: str = 'realsense',
        table_size: List[float] = None,
        device: str = 'cuda:0',
        visualize: bool = False,
        grasp_num: int = 0,
        num_samples: int = 4096,
        output_path: str = '',
        with_graspness: bool = False,
        load_gt_ibs: bool = False,
        scene_base_path: str = 'data/DexGraspNet2.0/scenes',
        mesh_base_path: str = 'data/DexGraspNet2.0/meshdata',
        **kwargs
    ):
        # Support both new-style kwargs and old-style DictConfig
        if hasattr(scene_id, 'scene_id'):
            # Old-style: scene_id is actually a cfg object
            cfg = scene_id
            self._init_from_config(cfg)
        else:
            # New-style: direct parameters
            self.robot_name = robot_name
            self.urdf_path = urdf_path
            self.meta_path = meta_path
            self.camera = camera
            self.scene_id = int(scene_id) if isinstance(scene_id, str) and scene_id.startswith('scene_') else int(scene_id) if isinstance(scene_id, str) else scene_id
            if isinstance(scene_id, str) and scene_id.startswith('scene_'):
                self.scene_id = int(scene_id.replace('scene_', ''))
            self.scene_name = 'scene_' + str(self.scene_id).zfill(4)
            self.grasp_num = grasp_num
            self.output_path = output_path
            self.with_graspness = with_graspness
            self.visualize = visualize
            self.table_size = table_size if table_size is not None else [0.6, 0.6, 0.0]
            self.num_samples = num_samples
            self.load_gt_ibs = load_gt_ibs
            self.device = device
            self.scene_base_path = scene_base_path
            self.mesh_base_path = mesh_base_path
        
        # Initialize scene
        self.calculate_mesh_pc()
        self.get_camera_pose(0)
        
        if self.visualize:
            self.vis = Vis(
                robot_name=self.robot_name,
                urdf_path=self.urdf_path,
                meta_path=self.meta_path
            )
            self.robot_plotly = []
            self.pc_plotly = []
            self.coord_plotly = []
        
        if self.grasp_num == 0:
            cprint(f"Set not to load grasps from downloaded dataset", 'yellow')
        else:
            self.load_all_grasps()
        
        if self.load_gt_ibs:
            self.load_ibs()
    
    def _init_from_config(self, cfg):
        """Initialize from DictConfig (backward compatibility)."""
        self.robot_name = getattr(cfg, 'robot_name', 'leap_hand')
        self.urdf_path = getattr(cfg, 'urdf_path', 'robot_models/urdf/leap_hand_simplified.urdf')
        self.meta_path = getattr(cfg, 'meta_path', 'robot_models/meta/leap_hand/meta.yaml')
        self.camera = getattr(cfg, 'camera', 'realsense')
        self.scene_id = cfg.scene_id
        self.scene_name = 'scene_' + str(self.scene_id).zfill(4)
        self.grasp_num = getattr(cfg, 'grasp_num', 0)
        self.output_path = getattr(cfg, 'output_path', '')
        self.with_graspness = getattr(cfg, 'with_graspness', False)
        self.visualize = getattr(cfg, 'visualize', False)
        self.table_size = list(getattr(cfg, 'table_size', [0.6, 0.6, 0.0]))
        self.num_samples = getattr(cfg, 'num_samples', 4096)
        self.load_gt_ibs = getattr(cfg, 'load_gt_ibs', False)
        self.device = getattr(cfg, 'device', 'cuda:0')
        self.scene_base_path = getattr(cfg, 'scene_base_path', 'data/DexGraspNet2.0/scenes')
        self.mesh_base_path = getattr(cfg, 'mesh_base_path', 'data/DexGraspNet2.0/meshdata')

    def calculate_mesh_pc(self):
        """Calculate mesh point clouds for all objects in the scene."""
        scene_path = os.path.join(self.scene_base_path, self.scene_name)
        extrinsics_path = os.path.join(scene_path, 'realsense/cam0_wrt_table.npy')
        extrinsics = np.load(extrinsics_path)
        annotation_path = os.path.join(scene_path, 'realsense/annotations/0000.xml')
        annotation = ET.parse(annotation_path)
        object_pose_dict = {}
        for obj in annotation.findall('obj'):
            object_code = str(int(obj.find('obj_id').text)).zfill(3)
            translation = np.array([float(x) for x in obj.find('pos_in_world').text.split()])
            rotation = np.array([float(x) for x in obj.find('ori_in_world').text.split()])
            rotation = transforms3d.quaternions.quat2mat(rotation)
            object_pose_dict[object_code] = dict(
                translation=extrinsics[:3, :3] @ translation + extrinsics[:3, 3],
                rotation=extrinsics[:3, :3] @ rotation, 
            )
        object_pose_dict = dict(sorted(object_pose_dict.items()))
        self.object_pose_dict = object_pose_dict
        self.object_meshes = {}
        for object_code in object_pose_dict.keys():
            object_path = os.path.join(self.mesh_base_path, object_code, 'simplified.obj')
            obj_mesh = tm.load(object_path)
            object_translation = object_pose_dict[object_code]['translation']
            object_rotation = object_pose_dict[object_code]['rotation']
            object_transform = np.eye(4)
            object_transform[:3,3] = object_translation
            object_transform[:3,:3] = object_rotation
            self.object_meshes[object_code] = obj_mesh.apply_transform(object_transform)
        
        # get object sampled points respectively
        num_objects = len(self.object_meshes.values())
        self.object_points_list = []
        for object_mesh in self.object_meshes.values():
            obj_vertices = torch.tensor(object_mesh.vertices, dtype=torch.float, device=self.device)
            obj_faces = torch.tensor(object_mesh.faces, dtype=torch.float, device=self.device)
            mesh = pytorch3d.structures.Meshes(obj_vertices.unsqueeze(0), obj_faces.unsqueeze(0))
            obj_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=(self.num_samples*2//num_objects))
            self.object_points_list.append(obj_point_cloud)

        self.combined_mesh = tm.util.concatenate(self.object_meshes.values())
        vertices = torch.tensor(self.combined_mesh.vertices, dtype=torch.float, device=self.device)
        faces = torch.tensor(self.combined_mesh.faces, dtype=torch.float, device=self.device)
        mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=20 * self.num_samples)
        obj_surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
        
        x = torch.arange(-self.table_size[0] / 2, self.table_size[0] / 2, 0.01)
        y = torch.arange(-self.table_size[1] / 2, self.table_size[1] / 2, 0.01)
        xn = x.shape[0]
        yn = y.shape[0]
        x = x.repeat(yn).to(self.device)
        y = y.repeat_interleave(xn).to(self.device)
        z = torch.full_like(x, self.table_size[2], device=self.device)
        table_surface_points = torch.stack([x, y, z], dim=1)

        self.surface_points = torch.cat([obj_surface_points, table_surface_points], dim=0)

    def get_plotly_mesh(self, size=3, opacity=0.8, transform=None, color='lightgreen'):
        mesh = deepcopy(self.combined_mesh)
        if transform is not None:
            mesh.apply_transform(transform)
        vertices = np.asarray(mesh.vertices)  
        faces = np.asarray(mesh.faces)       
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        i = faces[:, 0]
        j = faces[:, 1]
        k = faces[:, 2]
        mesh3d = go.Mesh3d(
            x=x,y=y,z=z,i=i,j=j,k=k,
            color=color,
            opacity=opacity,
            name='scene_mesh',
            flatshading=True  # Enable flat shading for better visual effect
        )
        return mesh3d

    def get_camera_pose(self, view):
        """Get camera pose for a specific view."""
        self.cam0_wrt_table = np.load(os.path.join(self.scene_base_path, self.scene_name, self.camera, 'cam0_wrt_table.npy'))
        self.camera_pose_wrt_cam0 = np.load(os.path.join(self.scene_base_path, self.scene_name, self.camera, 'camera_poses.npy'))[view]
        self.camera_pose = torch.from_numpy(np.einsum('ab,bc->ac', self.cam0_wrt_table, self.camera_pose_wrt_cam0)).float()

    def get_view_pc(self, view, downsample_pc_n=100000):
        view_plotly, pc, extrinsics = self.vis.scene_plotly(self.scene_name, view, self.camera, with_pc=True, mode='pc', graspness_path='dex_graspness_new' if self.with_graspness else None, with_extrinsics=True)
        idxs = torch.randperm(len(pc))[:downsample_pc_n]
        if not self.with_graspness:
            return pc.float()[idxs, :3], extrinsics
        raw_graspness = (pc[:, 4] + 1e-3).log()
        objectness = (pc[:, 3] != 0)
        graspness = torch.where(objectness, raw_graspness, raw_graspness * 0 + np.log(1e-3))
        return pc.float()[idxs, :3], extrinsics, graspness[idxs]
    
    # def get_precalculated_pc():
    #     scene_data = np.load(os.path.join('data', 'scenes', 'scene_' + str(scene_id).zfill(4), 'realsense', 'network_input.npz'))
    #     scene_pc_c = scene_data['pc'][view_id]  # (N, 3)
    #     scene_pc_c = torch.from_numpy(scene_pc_c).float()
    #     c2w_trans = scene_data['extrinsics'][view_id]
    #     c2w_trans = torch.from_numpy(c2w_trans).float()
    #     scene_pc_w = transform_points(scene_pc_c, c2w_trans)  # (N, 3), world coordinate system
    #     scene_pc_h = batch_transform_points(scene_pc_w.unsqueeze(0).repeat(self.ibs_load_per_scene, 1, 1), torch.from_numpy(w2h_trans).float())


    def get_mesh_plotly(self, transform=None, color='lightgreen'):
        self.scene_name_mesh_data_plotly = []
        for object_code, object_mesh in self.object_meshes.items():
            if transform is not None:
                object_mesh = deepcopy(object_mesh)
                object_mesh.apply_transform(transform)
            vertices = object_mesh.vertices
            faces = object_mesh.faces
            mesh = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0], 
                j=faces[:, 1], 
                k=faces[:, 2],
                color=color, 
                opacity=1, 
                hoverinfo='text',
                text=[object_code] * len(faces),
            )
            self.scene_name_mesh_data_plotly.append(mesh)
        return self.scene_name_mesh_data_plotly
    
    def load_all_grasps(self, view=0):
        path = os.path.join('data', 'dex_grasps_new', self.scene_name, self.robot_name)
        cprint(f"Loading grasps from {path}", "light_yellow")
        self.grasp_data = {}
        self.grasp_data_num = {}
        for p in os.listdir(path):
            if not p.endswith('.npz'):
                continue
            data = np.load(os.path.join(path, p))
            grasp_num = len(data['point'])
            if self.grasp_num is not None and self.grasp_num < grasp_num:
                grasp_num = self.grasp_num
                idxs = np.random.randint(0, len(data['point']), self.grasp_num)
                data = {k: data[k][idxs] for k in data.files}
                cprint(f"downsampled grasps for obj:{p} in {self.scene_name}, selected {self.grasp_num} from a total of {len(data['point'])}", "yellow")
            else:
                cprint(f"loaded grasps for obj:{p} in {self.scene_name} for a total of {grasp_num}", "yellow")

            # Convert data to PyTorch tensors
            points = torch.from_numpy(data['point']).float()  # [N, 3]
            translations = torch.from_numpy(data['translation']).float()  # [N, 3]
            rotations = torch.from_numpy(data['rotation']).float()  # [N, 3, 3]
            qpos = {k: torch.from_numpy(data[k]).float() for k in data.keys() if k not in ['point', 'translation', 'rotation']}  # Joint angles

            # Transform to camera frame (relative to cam0)
            cam0_rot = torch.from_numpy(self.camera_pose_wrt_cam0[:3, :3]).float()
            cam0_trans = torch.from_numpy(self.camera_pose_wrt_cam0[:3, 3]).float()
            
            # Points: Rotate and subtract cam0 translation
            points_cam = torch.einsum('ab,nb->na', cam0_rot, points - cam0_trans)  # [N, 3]
            # Translations: Rotate and subtract cam0 translation
            trans_cam = torch.einsum('ab,nb->na', cam0_rot, translations - cam0_trans)  # [N, 3]
            # Rotations: Rotate
            rot_cam = torch.einsum('ab,nbc->nac', cam0_rot, rotations)  # [N, 3, 3]

            # Transform to world frame
            world_rot = self.camera_pose[:3, :3]
            world_trans = self.camera_pose[:3, 3]
            
            # Points: Rotate and add world translation
            points_world = torch.einsum('ab,nb->na', world_rot, points_cam) + world_trans  # [N, 3]
            # Translations: Rotate and add world translation
            trans_world = torch.einsum('ab,nb->na', world_rot, trans_cam) + world_trans  # [N, 3]
            # Rotations: Rotate
            rot_world = torch.einsum('ab,nbc->nac', world_rot, rot_cam)  # [N, 3, 3]

            # Store transformed data
            self.grasp_data[p] = {
                'gp': points_world,  # World frame points
                'trans': trans_world,  # World frame translations
                'rot': rot_world,  # World frame rotations
                'qpos': qpos  # Joint angles (unchanged)
            }
            self.grasp_data_num[p] = grasp_num

            # Visualization (if enabled)
            if self.visualize:
                for i in range(grasp_num):
                    trans_i = trans_world[i:i+1]  # [1, 3]
                    rot_i = rot_world[i:i+1]  # [1, 3, 3]
                    point_i = points_world[i:i+1]  # [1, 3]
                    qpos_i = {k: v[i:i+1] for k, v in qpos.items()}  # Slice joint angles
                    self.robot_plotly += self.vis.robot_plotly(trans_i, rot_i, qpos_i, opacity=0.5, color=random.choice(px.colors.sequential.Plasma))
                    self.pc_plotly += self.vis.pc_plotly(point_i, size=5, color='red')
                    self.coord_plotly += self.vis.coordinate_frame_plotly(point_i[0], rot_i[0])
        cprint(f"{self.scene_name} grasp data num: {self.grasp_data_num}", "green")

    def load_ibs(self):
        ibs_data_path = 'data/ibsdata'
        ibs_p = os.path.join(ibs_data_path, 'ibs', f"scene_{str(self.scene_id).zfill(4)}.npy")
        w2h_trans_p = os.path.join(ibs_data_path, 'w2h_trans', f"scene_{str(self.scene_id).zfill(4)}.npy")
        self.ibs = np.load(ibs_p)
        self.w2h_trans = np.load(w2h_trans_p)
        self.ibs_data_num = self.ibs.shape[0]

    def visualize_plotly(self, view=0):
        pc, extrinsics, graspness = self.get_view_pc(view)
        pc = torch.einsum('ab,nb->na', self.camera_pose[:3, :3], pc) + self.camera_pose[:3, 3]  # Transform from camera to world coordinates
        view_plotly = self.vis.pc_plotly(pc, size=1, value=graspness)

        plotly = view_plotly + self.robot_plotly + self.pc_plotly + self.coord_plotly + self.get_mesh_plotly()

        self.vis.show(plotly, self.output_path)

    def get_target_object_points_from_grasp_points(self, grasp_points):
        """
        Find the closest points from object points, determine which object is target object and return its sampled point cloud.
        Args:
            grasp_points: numpy.ndarray, (B, 3)
        Returns:
            target_object_points: torch.Tensor, (B, N, 3)
        """
        # Convert inputs to torch tensors if not already
        grasp_points = torch.from_numpy(grasp_points).float().to(self.device)  # (B, 3)
        
        # Initialize variables
        B = grasp_points.shape[0]
        N = self.object_points_list[0].shape[1]  # Number of points per object
        min_distances = torch.full((B,), float('inf')).to(self.device)
        target_indices = torch.zeros(B, dtype=torch.long).to(self.device)
        
        # Iterate through all objects to find closest points
        for obj_idx, obj_points in enumerate(self.object_points_list):
            # obj_points: (N, 3)
            obj_points = obj_points.to(self.device).float()
            
            # Compute distances between grasp points and object points
            # Expand dimensions to compute pairwise distances
            grasp_points_exp = grasp_points.unsqueeze(1)  # (B, 1, 3)
            obj_points_exp = obj_points    # (1, N, 3)
            distances = torch.norm(grasp_points_exp - obj_points_exp, dim=2)  # (B, N)
            
            # Find minimum distance for each grasp point
            min_dist, _ = distances.min(dim=1)  # (B,)
            
            # Update target indices where this object is closer
            mask = min_dist < min_distances
            min_distances[mask] = min_dist[mask]
            target_indices[mask] = obj_idx
        
        # Gather target object points for each batch
        target_object_points = torch.zeros(B, N, 3).to(self.device)
        for b in range(B):
            target_object_points[b] = self.object_points_list[target_indices[b]][0]
        
        return target_object_points


def main():
    """Main entry point for testing Scene class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Scene class')
    parser.add_argument('--scene_id', type=int, default=55, help='Scene ID')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    scene = Scene(
        scene_id=args.scene_id,
        device=args.device,
        visualize=args.visualize
    )
    
    if args.visualize:
        scene.visualize_plotly()
    
    print(f"Scene {scene.scene_name} loaded successfully")
    print(f"  - Objects: {list(scene.object_meshes.keys())}")
    print(f"  - Surface points: {scene.surface_points.shape}")


if __name__ == '__main__':
    main()