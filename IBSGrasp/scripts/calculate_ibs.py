import os
import sys
import argparse
import open3d as o3d
import trimesh as tm
import hydra
import yaml
from omegaconf import DictConfig

import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from termcolor import cprint
import plotly.express as px

import torch
import transforms3d
from pytorch3d.ops import sample_farthest_points

sys.path.append('.')
from IBSGrasp.utils.transforms import batch_transform_points, transform_points
from IBSGrasp.scripts.scene import Scene
from src.utils.robot_model import RobotModel
from src.utils.vis_plotly import Vis

import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from typing import List
from matplotlib.colors import to_rgb

def parse_plotly_color(color):
    """Convert Plotly color to Open3D RGB format ([0, 1], [0, 1], [0, 1])."""
    if isinstance(color, str):
        from plotly.colors import named_colorscales, unlabel_rgb
        if color.startswith('#'):
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        else:
            try:
                rgb = to_rgb(color)
            except KeyError:
                print(f"Unknown color {color}, defaulting to gray")
                rgb = (0.5, 0.5, 0.5)
    elif isinstance(color, (list, tuple, np.ndarray)):
        rgb = np.array(color)[:3] / 255.0 if np.max(color) > 1 else np.array(color)[:3]
    else:
        rgb = (0.5, 0.5, 0.5)
    return np.array(rgb)

def o3d_show(plotly_list: List[go.Figure]):
    """
    Visualize Plotly data (point clouds and meshes) using Open3D, dynamically extracting point_size from Plotly data.

    Args:
        plotly_list: List of go.Scatter3d (point clouds) and go.Mesh3d (meshes).
    """
    geometries = []
    point_sizes = []  # Collect point sizes from all point clouds

    for item in plotly_list:
        if isinstance(item, go.Scatter3d):
            # Handle point cloud (go.Scatter3d)
            points = np.vstack([item.x, item.y, item.z]).T
            if len(points) == 0:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Extract color
            marker = item.marker if hasattr(item, 'marker') else None
            if marker and hasattr(marker, 'color'):
                color = parse_plotly_color(marker.color)
                colors = np.tile(color, (len(points), 1))
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), 0.5))

            # Extract point size
            if marker and hasattr(marker, 'size'):
                size = marker.size
                if isinstance(size, (int, float)):
                    point_sizes.append(size)
                elif isinstance(size, (list, np.ndarray)):
                    point_sizes.append(np.mean(size))
                else:
                    point_sizes.append(2.0)  # Default size
            else:
                point_sizes.append(2.0)  # Default size

            geometries.append(pcd)

        elif isinstance(item, go.Mesh3d):
            # Handle mesh (go.Mesh3d)
            vertices = np.vstack([item.x, item.y, item.z]).T
            triangles = np.vstack([item.i, item.j, item.k]).T

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(list(to_rgb(item.color)))
            geometries.append(mesh)

    if not geometries:
        print("No valid geometries to visualize")
        return

    # Calculate average point size
    avg_point_size = np.mean(point_sizes) if point_sizes else 2.0

    # Set up visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geometries:
        vis.add_geometry(geom)

    # render_option = vis.get_render_option()
    # render_option.point_size = avg_point_size

    vis.run()
    vis.destroy_window()

def add_transform_perturbation(transforms, max_translation=1.5, max_rotation=5):
    """
    为变换矩阵添加平移和旋转的随机扰动。
    
    参数:
        transforms (np.ndarray): 形状为 (N, 4, 4) 的变换矩阵。
        max_translation (float): 平移扰动的最大值（单位：cm）。
        max_rotation (float): 旋转扰动的最大值（单位：度）。
    
    返回:
        np.ndarray: 添加扰动后的变换矩阵。
    """
    N = transforms.shape[0]
    
    # 1. 生成平移扰动
    translation_perturbation = np.random.normal(0, max_translation / 3, (N, 3))  # 正态分布，标准差为 max_translation / 3
    translation_perturbation = np.clip(translation_perturbation, -max_translation, max_translation)  # 限制在 [-max_translation, max_translation] 内
    
    # 将平移扰动转换为齐次变换矩阵
    translation_matrices = np.eye(4)[np.newaxis, :, :].repeat(N, axis=0)
    translation_matrices[:, :3, 3] = translation_perturbation
    
    # 2. 生成旋转扰动
    rotation_perturbation = np.random.normal(0, max_rotation / 3, (N, 3))  # 正态分布，标准差为 max_rotation / 3
    rotation_perturbation = np.clip(rotation_perturbation, -max_rotation, max_rotation)  # 限制在 [-max_rotation, max_rotation] 内
    
    # 将旋转扰动转换为旋转矩阵
    rotation_matrices = np.zeros((N, 4, 4))
    for i in range(N):
        # 将欧拉角转换为旋转矩阵
        rx, ry, rz = np.radians(rotation_perturbation[i])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx  # 组合旋转矩阵
        rotation_matrices[i, :3, :3] = R
        rotation_matrices[i, 3, 3] = 1
    
    # 3. 组合平移和旋转扰动
    perturbation_matrices = translation_matrices @ rotation_matrices
    
    # 4. 将扰动应用到原始变换矩阵
    perturbed_transforms = perturbation_matrices @ transforms
    
    return perturbed_transforms


def calculate_IBS(cfg):

    ibs_save_path = cfg.ibs_save_path
    if not cfg.make_picture:
        os.makedirs(os.path.join(ibs_save_path, 'ibs'),exist_ok=True)
        os.makedirs(os.path.join(ibs_save_path, 'w2h_trans'),exist_ok=True)
        os.makedirs(os.path.join(ibs_save_path, 'hand_dis'),exist_ok=True)
        if os.path.exists(os.path.join(ibs_save_path, 'ibs', f'scene_{str(cfg.scene_id).zfill(4)}.npy')):
            return

    pose_data_path = cfg.pose_data_path
    scene_cfg_path = cfg.scene_cfg_path
    scene_id = cfg.scene_id
    cprint(f"Processing {scene_id}", "green")
    grasp_data = np.load(os.path.join(pose_data_path, 'scene_'+str(scene_id).zfill(4), 'success_grasps.npz'), allow_pickle=True)
    grasp_data = grasp_data['arr_0'].tolist()
    scene_cfg = DictConfig(yaml.safe_load(open(scene_cfg_path,'r')))
    scene_cfg.grasp_num = 0
    scene_cfg.visualize = False
    scene_cfg.scene_id = scene_id
    scene_cfg.device = cfg.device
    scene = Scene(scene_cfg)
    
    batch_size = grasp_data['translation'].shape[0]
    if batch_size > cfg.max_ibs_per_scene:
        # selected_idxs = np.random.randint(0, batch_size, cfg.max_ibs_per_scene)
        # FPS select
        gps = grasp_data['grasppoints']
        gps = torch.tensor(gps).unsqueeze(0)
        gps += torch.rand(gps.shape, device=gps.device, dtype=gps.dtype)*0.02-0.01
        _, fps_idxs = sample_farthest_points(gps, K=cfg.max_ibs_per_scene)
        selected_idxs = fps_idxs[0].cpu().numpy()
        batch_size = cfg.max_ibs_per_scene
    else:
        selected_idxs = range(batch_size)
    if batch_size == 0:
        return
    robot_model = RobotModel(
        urdf_path='robot_models/urdf/leap_hand_simplified.urdf',
        meta_path='robot_models/meta/leap_hand/meta.yaml'
    )
    trans = grasp_data['translation'][selected_idxs]
    rot = grasp_data['rotation'][selected_idxs]
    gp = grasp_data['grasppoints'][selected_idxs]
    qpos_dict = {k:torch.from_numpy(v).to(device=cfg.device)[selected_idxs] for k,v in grasp_data.items() if k not in ['translation', 'rotation', 'grasppoints']}
    hand_thumb_points, _, hand_whole_points = \
        robot_model.get_finger_surface_points(
            torch.from_numpy(trans).to(device=cfg.device), 
            torch.from_numpy(rot).to(device=cfg.device), 
            qpos_dict, n_points_each_link=300)
    local_frame = np.eye(4)[None].repeat(batch_size, axis=0)
    local_frame[:,:3,3] = gp
    local_frame[:,:3,:3] = rot
    world_to_hand_coord_transforms = torch.from_numpy(np.linalg.inv(local_frame)).to(dtype=torch.float, device=cfg.device)
    scene_surface_points = scene.surface_points
    target_object_points = scene.get_target_object_points_from_grasp_points(gp)
    hand_surface_points = batch_transform_points(hand_whole_points, world_to_hand_coord_transforms)
    hand_thumb_points = batch_transform_points(hand_thumb_points, world_to_hand_coord_transforms)
    scene_surface_points = batch_transform_points(scene_surface_points.unsqueeze(0).repeat(batch_size,1,1), world_to_hand_coord_transforms)
    print(target_object_points.shape, world_to_hand_coord_transforms.shape)
    target_object_points = batch_transform_points(target_object_points, world_to_hand_coord_transforms)
    bound = 0.1
    resolution = 0.005
    delta = 0.005
    epsilon = 1e-5
    max_iteration = 20
    r = 0.1
    grid_x, grid_y, grid_z = np.mgrid[-bound:bound:resolution, -bound:bound:resolution, -bound:bound:resolution]
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    batch_points = points[np.newaxis, :, :].repeat(batch_size, axis=0)
    batch_points_tensor = torch.tensor(batch_points, dtype=torch.float, device=cfg.device)
    
    def points_query(pc, points):
        distances = torch.cdist(points, pc)
        min_distances, min_indices = torch.min(distances, dim=1)
        nearest_points = pc[min_indices]
        vectors = nearest_points - points
        unit_vectors = vectors / (torch.norm(vectors, dim=1, keepdim=True)+1e-4)
        return min_distances, unit_vectors, min_indices

    ibs_con_voxels = []
    hand_dis_voxels = []

    for grasp_id in tqdm(range(batch_size), desc=f"iteration for {scene_id}"):
        scene_dis, _, _ = points_query(scene_surface_points[grasp_id], batch_points_tensor[grasp_id])
        target_obj_dis, _, _ = points_query(target_object_points[grasp_id], batch_points_tensor[grasp_id])
        hand_dis, _, _ = points_query(hand_surface_points[grasp_id], batch_points_tensor[grasp_id])
        thumb_dis, _, _ = points_query(hand_thumb_points[grasp_id], batch_points_tensor[grasp_id])

        if cfg.pc2vox:
            ibs_mask = torch.abs(scene_dis - hand_dis) < delta * 2
            ibs_points = batch_points_tensor[grasp_id][ibs_mask]
            for i in range(max_iteration):
                scene_dis_, scene_uv, _ = points_query(scene_surface_points[grasp_id], ibs_points)
                hand_dis_, hand_uv, _ = points_query(hand_surface_points[grasp_id], ibs_points)
                dis_diff = scene_dis_ - hand_dis_
                if torch.max(torch.abs(dis_diff)) < epsilon:
                    break
                adjustments = torch.where(dis_diff[:, None] > 0, scene_uv, -hand_uv)
                ibs_points = ibs_points + torch.clamp((dis_diff / (1 - torch.sum(scene_uv*hand_uv,dim=1))),min=-0.01,max=0.01)[:, None] * adjustments
            ibs_points = (ibs_points - torch.tensor([-bound, -bound, -bound], device=cfg.device)) / resolution
            ibs_points = torch.floor(ibs_points).long()
            ibs_points = torch.clamp(ibs_points, 0, 39)
            ibs_mask = torch.zeros((40,40,40), dtype=torch.bool, device=cfg.device)
            ibs_mask[ibs_points[:, 0], ibs_points[:, 1], ibs_points[:, 2]] = 1
            ibs_mask = ibs_mask.ravel()
        else:
            ibs_mask = torch.abs(scene_dis - hand_dis) < delta

        contact_mask = (target_obj_dis < delta*1.5) & (hand_dis < delta*1.5) & ibs_mask
        thumb_contact_mask = contact_mask & (thumb_dis < delta*1.7)
        under_table_mask = transform_points(batch_points_tensor[grasp_id],torch.inverse(world_to_hand_coord_transforms[grasp_id]))[:,2]<scene.table_size[2]
        ibs_mask = ibs_mask & ~under_table_mask
        ibs_mask = ibs_mask.reshape(40,40,40,1)
        contact_mask = contact_mask.reshape(40,40,40,1)
        thumb_contact_mask = thumb_contact_mask.reshape(40,40,40,1)
        ibs_con_voxels.append(torch.cat([ibs_mask,contact_mask,thumb_contact_mask],dim=3))
        hand_dis_voxels.append(hand_dis.reshape(40,40,40))


        if cfg.vis and grasp_id<5:
            colors = px.colors.sequential.Plasma
            ibs_mask = ibs_mask.ravel()
            contact_mask = contact_mask.ravel()
            thumb_contact_mask = thumb_contact_mask.ravel()
            # ibs_mask = contact_mask & ibs_mask
            ibs_points = batch_points_tensor[grasp_id][ibs_mask&~contact_mask]
            contact_points = batch_points_tensor[grasp_id][contact_mask&~thumb_contact_mask].cpu()
            thumb_contact_points = batch_points_tensor[grasp_id][thumb_contact_mask].cpu()
            scene_dis = scene_dis[ibs_mask].cpu()
            hand_dis = hand_dis[ibs_mask].cpu()

            sampled_ibs_points = ibs_points.cpu()
            vis = Vis(robot_name=scene.robot_name,
                      urdf_path=scene.urdf_path,
                      meta_path=scene.meta_path)

            contact_plotly = vis.pc_plotly(contact_points, color='red',size=10)
            thumb_plotly = vis.pc_plotly(thumb_contact_points, color='blue',size=10)
            ibs_plotly = vis.pc_plotly(sampled_ibs_points, color=colors[6],size=6)
            scene_plotly = scene.get_mesh_plotly(world_to_hand_coord_transforms[grasp_id].cpu().numpy(), 'green')
            scene_pc_plotly = vis.pc_plotly(scene_surface_points[grasp_id].cpu(), color=colors[3], size=3)
            target_pc_plotly = vis.pc_plotly(target_object_points[grasp_id].cpu(), color=colors[4], size=3)
            hand_plotly = vis.pc_plotly(hand_surface_points[grasp_id].cpu(), color=colors[5], size=3)
            trans_w = trans[grasp_id]
            rot_w = rot[grasp_id]
            transform_w = np.eye(4)
            transform_w[:3,3]=trans_w
            transform_w[:3,:3]=rot_w
            transform_i = world_to_hand_coord_transforms[grasp_id].cpu().numpy() @ transform_w
            trans_i = torch.from_numpy(transform_i[:3,3][None]).float().to(device='cuda:0')
            rot_i = torch.from_numpy(transform_i[:3,:3][None]).float().to(device='cuda:0')
            qpos_dict_ibs = {k:v[grasp_id:grasp_id+1] for k,v in qpos_dict.items()}
            hand_mesh_plotly = vis.robot_plotly(
                trans_i,
                rot_i,
                qpos_dict_ibs,
                color=(colors[8],colors[9],colors[7])
            )
            coord_plotly = vis.coordinate_frame_plotly()
            # vis.show(contact_plotly+thumb_plotly+scene_plotly+hand_plotly+hand_mesh_plotly+ibs_plotly+coord_plotly+scene_pc_plotly+target_pc_plotly)
            # vis.show(contact_plotly+thumb_plotly+scene_plotly+hand_mesh_plotly+ibs_plotly)
            o3d_show(contact_plotly+thumb_plotly+scene_plotly+hand_mesh_plotly+ibs_plotly)
            # vis.show(contact_plotly+thumb_plotly+ibs_plotly+target_pc_plotly)
            if grasp_id==5:
                break

    torch.cuda.empty_cache()
    if not cfg.make_picture:
        np.save(os.path.join(ibs_save_path, 'ibs', scene.scene_name+'.npy'), torch.stack(ibs_con_voxels).cpu().numpy())
        np.save(os.path.join(ibs_save_path, 'w2h_trans', scene.scene_name+'.npy'), world_to_hand_coord_transforms.cpu().numpy())
        np.save(os.path.join(ibs_save_path, 'hand_dis', scene.scene_name+'.npy'), torch.stack(hand_dis_voxels).cpu().numpy())

@hydra.main(version_base="v1.2", config_path='../conf', config_name='ibs')
def main(cfg: DictConfig):
    calculate_IBS(cfg)

if __name__ == '__main__':
    main()