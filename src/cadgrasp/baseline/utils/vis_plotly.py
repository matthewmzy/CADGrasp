import os
import sys

from typing import Optional, Union, Dict
import trimesh as tm
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.io as scio
import torch
import plotly.graph_objects as go
import plotly.express as px
import plotly.express as px
import random

from cadgrasp.paths import project_path
from cadgrasp.baseline.utils.robot_model import RobotModel
from cadgrasp.baseline.utils.ik import IK
from cadgrasp.baseline.utils.rot6d import compute_ortho6d_from_rotation_matrix
from cadgrasp.baseline.utils.pc import depth_image_to_point_cloud, get_workspace_mask
from cadgrasp.baseline.utils.robot_info import GRIPPER_HEIGHT, GRIPPER_FINGER_WIDTH, GRIPPER_TAIL_LENGTH, GRIPPER_DEPTH_BASE

class Vis:
    def __init__(self, 
                 robot_name: str = 'leap_hand',
                 urdf_path: Optional[str] = None, 
                 meta_path: Optional[str] = None,
    ):
        self.robot_name = robot_name
        if self.robot_name == 'gripper':
            self.robot_joints = ['width', 'depth']
        else:
            urdf_path = project_path(f'robot_models/urdf/{robot_name}.urdf') if urdf_path is None else urdf_path
            meta_path = project_path(f'robot_models/meta/{robot_name}/meta.yaml') if meta_path is None else meta_path
            self.robot = RobotModel(
                urdf_path=urdf_path,
                meta_path=meta_path,
            )
            self.robot_joints = self.robot.joint_names
    
    def box_plotly(self,
                   scale: np.ndarray, # (3, )
                   trans: Optional[np.ndarray] = None, # (3, )
                   rot: Optional[np.ndarray] = None, # (3, 3)
                   opacity: Optional[float] = None,
                   color: Optional[str] = None,
        ) -> list:

        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity

        # 8 vertices of a cube
        corner = np.array([[0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1]]).T - 0.5
        corner *= scale
        corner = np.einsum('ij,kj->ki', rot, corner) + trans
        
        return [go.Mesh3d(
            x = corner[:, 0],
            y = corner[:, 1],
            z = corner[:, 2],
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color,
            opacity=opacity,
        )]
    
    def plane_plotly(self,
                     plane_vec: torch.Tensor, # (4, )
                     opacity: Optional[float] = None,
                     color: Optional[str] = None,
    ) -> list:

        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity

        dir = plane_vec[:3]
        assert (torch.linalg.norm(dir) - 1).abs() < 1e-4
        center = dir * -plane_vec[3]
        z_axis = dir
        x_axis = torch.zeros(3)
        x_axis[dir.abs().argmin()] = 1
        x_axis = x_axis - (x_axis * z_axis).sum() * z_axis
        x_axis = x_axis / torch.linalg.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis)
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        return self.box_plotly(np.array([1,1,0]), center, rot, color='blue')
    
    def robot_plotly(self,
                     trans: Optional[torch.tensor] = None, # (1, 3)
                     rot: Optional[torch.tensor] = None, # (1, 3, 3)
                     qpos: Optional[Union[torch.tensor, Dict[str, torch.Tensor]]] = None, # (1, n) or Dict[str, (1,)]
                     opacity: Optional[float] = None,
                     color: Optional[str] = None,
                     mesh_type: str = 'collision',
    ) -> list:
        if trans is None:
            trans = torch.zeros(1, 3)
        if rot is None:
            rot = torch.eye(3)[None]
        if qpos is None:
            qpos = torch.zeros(1, len(self.robot_joints))
        if color is None:
            color = 'violet'
        if opacity is None:
            opacity = 1.0
        
        if self.robot_name == 'gripper':
            # changed for better visualization
            height = 0.002 # GRIPPER_HEIGHT
            finger_width = 0.002 # GRIPPER_FINGER_WIDTH
            tail_length = GRIPPER_TAIL_LENGTH
            depth_base = GRIPPER_DEPTH_BASE
            width, depth = qpos[0]
            """
            4 boxes: 
                       2|------ 1
                --------|  . O
                    3   |------ 0

                                        y
                                        | 
                                        O--x
                                       /
                                      z
            """
            centers = torch.tensor([[(depth - finger_width - depth_base)/2, (width + finger_width)/2, 0],
                                    [(depth - finger_width - depth_base)/2, -(width + finger_width)/2, 0],
                                    [-depth_base-finger_width/2, 0, 0],
                                    [-depth_base-finger_width-tail_length/2, 0, 0]])
            scales = torch.tensor([[finger_width+depth_base+depth, finger_width, height],
                                   [finger_width+depth_base+depth, finger_width, height],
                                   [finger_width, width, height],
                                   [tail_length, finger_width, height]])
            centers = torch.einsum('ij,kj->ki', rot[0], centers) + trans[0]
            box_plotly_list = []
            for i in range(4):
                box_plotly_list += self.box_plotly(scales[i].numpy(), centers[i].numpy(), rot[0].numpy(), opacity, color)
            return box_plotly_list
        else:
            # assume qpos in the RobotModel's joint order
            if type(qpos) == torch.Tensor:
                qpos_dict = {joint: qpos[:, i] for i, joint in enumerate(self.robot_joints)}
            else:
                qpos_dict = {joint: qpos[joint] for joint in self.robot_joints}
            link_translations, link_rotations = self.robot.forward_kinematics(qpos_dict)
            link_translations = {k: torch.einsum('nab,nb->na', rot, v) + trans for k, v in link_translations.items()}
            link_rotations = {k: torch.einsum('nab,nbc->nac', rot, v) for k, v in link_rotations.items()}
            plotly_data = []
            for link_name in link_translations.keys():
                vertices, faces = self.robot.get_link_mesh(link_name, mesh_type)
                plotly_data += self.mesh_plotly(vertices=vertices, faces=faces, trans=link_translations[link_name][0], rot=link_rotations[link_name][0], opacity=opacity, color=color)
            return plotly_data
    
    def pc_plotly(self, 
                  pc: torch.tensor, # (n, 3)
                  value: Optional[torch.tensor] = None, # (n, )
                  size: int = 1,
                  color: Union[str, torch.tensor] = 'red',
                  color_map: str = 'Viridis',
    ) -> list:
        if value is None: 
            if not isinstance(color, str):
                color = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in color.numpy()]
            pc_plotly = go.Scatter3d(
                x=pc[:, 0].numpy(),
                y=pc[:, 1].numpy(),
                z=pc[:, 2].numpy(),
                mode='markers',
                marker=dict(size=size, color=color),
            )
        else:
            pc_plotly = go.Scatter3d(
                x=pc[:, 0].numpy(),
                y=pc[:, 1].numpy(),
                z=pc[:, 2].numpy(),
                mode='markers',
                marker=dict(size=size, color=value.numpy(), colorscale=color_map, showscale=True),
            )
        return [pc_plotly]
    
    def line_plotly(self,
                    pc: torch.tensor, # (n, 3)
                    idxs: torch.tensor = None, # (n)
                    width: int = None,
                    color: str = None,
    ) -> list:
        color = 'green' if color is None else color
        width = 1 if width is None else width

        if idxs is None:
            idxs = torch.arange(len(pc))
        argsort = torch.argsort(idxs)
        pc = pc[argsort]
        x, y, z = pc[:, 0].numpy(), pc[:, 1].numpy(), pc[:, 2].numpy()
        return [go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(width=width, color=color),
        )]
    
    def mesh_plotly(self,
                    path: str = None,
                    scale: float = 1.0,
                    trans: Optional[torch.tensor] = None, # (3, )
                    rot: Optional[torch.tensor] = None, # (3, 3)
                    opacity: float = 1.0,
                    color: str = 'lightgreen',
                    vertices: Optional[torch.tensor] = None, # (n, 3)
                    faces: Optional[torch.tensor] = None, # (m, 3)
    ) -> list:
        if trans is None:
            trans = torch.zeros(3)
        if rot is None:
            rot = torch.eye(3)
        
        if path is not None:
            mesh = tm.load(path).apply_scale(scale)
            vertices, faces = mesh.vertices, mesh.faces
        else:
            vertices = vertices.numpy() * scale
            faces = faces.numpy()

        v = np.einsum('ij,kj->ki', rot.numpy(), vertices) + trans.numpy()
        f = faces
        mesh_plotly = go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            color=color,
            opacity=opacity,
        )
        return [mesh_plotly]
    
    def acronym_scene_plotly(self,
                             scene: str,
                             view: str,
                             camera: str = 'realsense',
                             mode: str = 'model',
                             opacity: Optional[float] = None,):
        path = f'data/scenes/scene_0000/{camera}'
        align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
        camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
        camera_pose = np.matmul(align_mat, camera_poses[int(view)])
        if mode == 'model':
            poses = np.load(os.path.join('data_acronym', 'scenes', scene+'.npz'))
            result = []
            for obj_idx, pose in poses.items():
                pose = torch.from_numpy(np.linalg.inv(camera_pose) @ pose).float()
                mesh_path = os.path.join('data_acronym', 'meshdata', obj_idx, 'scaled.obj')
                result += self.mesh_plotly(mesh_path, trans=pose[:3, 3], rot=pose[:3, :3], opacity=opacity)
            
            table_mat = np.linalg.inv(camera_pose)
            table_trans = table_mat[:3, 3] - 0.05 * table_mat[:3, 2]
            result += self.box_plotly(np.array([1,1,0.1]), table_trans, table_mat[:3,:3], color='blue')
            return result
    
    def acronym_scene_plotly_test(self,
                             scene: str,
                             view: str,
                             camera: str = 'realsense',
                             mode: str = 'model',
                            num_points: int = 40000,
                            with_pc: bool = False,
                            with_extrinsics: bool = False,
                            graspness_path: Optional[str] = None,
                            opacity: Optional[float] = None,
                            ):
        path = f'data/scenes/scene_0000/{camera}'
        if mode == 'pc':
            # loading
            split = scene.split('_')[1]
            depth = np.array(Image.open(os.path.join(f'data/acronym_test_scenes/test_acronym_{split}_depth_gt', camera, scene, str(view).zfill(4) + '.png')))
            seg = np.array(Image.open(os.path.join(f'data/acronym_test_scenes/test_acronym_{split}_label_gt', camera, scene, str(view).zfill(4) + '.png')))
            edge_path = os.path.join(f'data/acronym_test_scenes/network_input_{split}', scene, camera, 'edge_gt', str(view).zfill(4) + '.png')
            if os.path.exists(edge_path):
                edge = np.array(Image.open(edge_path))
            else:
                print(f'edge not found: {edge_path}')
                edge = np.zeros_like(depth)
            meta = scio.loadmat(os.path.join(path, 'meta', '0000.mat'))
            camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
            instrincs = meta['intrinsic_matrix']
            instrincs[0,0] *= 0.25
            instrincs[1,1] *= 0.25
            instrincs[0,2] *= 0.25
            instrincs[1,2] *= 0.25
            factor_depth = meta['factor_depth']

            # mask pc
            cloud = depth_image_to_point_cloud(depth, instrincs, factor_depth)
            depth_mask = (depth > 0)
            trans = np.dot(align_mat, camera_poses[int(view)])
            workspace_mask = get_workspace_mask(cloud, seg, trans)
            mask = (depth_mask & workspace_mask)
            cloud = cloud[mask]
            seg = seg[mask]
            edge = edge[mask]
            idxs = np.random.choice(len(cloud), num_points, replace=True)
            cloud = cloud[idxs]
            seg = seg[idxs]
            edge = edge[idxs]
            if graspness_path is not None:
                graspness = np.load(os.path.join('data', graspness_path, scene, camera, str(view).zfill(4) + '.npy'))
                graspness = graspness.reshape(-1)
                graspness = graspness[idxs]

            result = []
            for i, idx in enumerate(np.unique(seg)):
                result += self.pc_plotly(torch.from_numpy(cloud[seg == idx]), size=1, color=px.colors.qualitative.Dark24[i])
            if with_pc:
                list = [cloud, seg[:, None]]
                if graspness_path is not None:
                    list.append(graspness[:, None])
                list.append(edge[:, None])
                if with_extrinsics:
                    return result, torch.from_numpy(np.concatenate(list, axis=-1)).float(), trans
                else:
                    return result, torch.from_numpy(np.concatenate(list, axis=-1)).float()
            return result, None
        elif mode == 'model':
            align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
            camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
            camera_pose = np.matmul(align_mat, camera_poses[int(view)])
            if 'varobj' in scene:
                poses = np.load(os.path.join('data/acronym_test_scenes/stable_scenes_less', scene+'.npz'), allow_pickle=True)['arr_0'][None][0]
            else:
                poses = np.load(os.path.join('data/acronym_test_scenes/stable_scenes_test', scene+'.npz'), allow_pickle=True)['arr_0'][None][0]
            result = []
            for obj_idx, pose in poses.items():
                pose = torch.from_numpy(np.linalg.inv(camera_pose) @ pose).float()
                mesh_path = os.path.join('data/acronym/meshes/models', obj_idx, 'scaled.obj')
                result += self.mesh_plotly(mesh_path, trans=pose[:3, 3], rot=pose[:3, :3], opacity=opacity)
            
            table_mat = np.linalg.inv(camera_pose)
            table_trans = table_mat[:3, 3] - 0.05 * table_mat[:3, 2]
            result += self.box_plotly(np.array([1,1,0.1]), table_trans, table_mat[:3,:3], color='blue')
            return result
    
    
    def scene_plotly(self,
                     scene: str,
                     view: str,
                     camera: str,
                     mode: str = 'model',
                     num_points: int = 40000,
                     with_pc: bool = False,
                     with_extrinsics: bool = False,
                     graspness_path: Optional[str] = None,
                     opacity: Optional[float] = None,
                     gt: int = 1,
    ):
        """
            if with_pc is True, return both pc and plotly
            pc: torch.tensor (N, 5) with (x, y, z, seg, graspness)
        """
        path = project_path('data', 'DexGraspNet2.0', 'scenes', scene, camera)
        gt_str = "_gt" * gt
        if mode == 'pc':
            # loading
            depth = np.array(Image.open(os.path.join(path, 'depth'+gt_str, str(view).zfill(4) + '.png')))
            edge_path = os.path.join(path, 'edge'+gt_str, str(view).zfill(4) + '.png')
            if os.path.exists(edge_path):
                edge = np.array(Image.open(edge_path))
            else:
                print(f'edge not found: {edge_path}')
                edge = np.zeros_like(depth)
            seg = np.array(Image.open(os.path.join(path, 'label'+gt_str, str(view).zfill(4) + '.png')))
            meta = scio.loadmat(os.path.join(path, 'meta', str(view).zfill(4) + '.mat'))
            camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
            instrincs = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']

            # mask pc
            cloud = depth_image_to_point_cloud(depth, instrincs, factor_depth)
            depth_mask = (depth > 0)
            trans = np.dot(align_mat, camera_poses[int(view)])
            workspace_mask = get_workspace_mask(cloud, seg, trans)
            mask = (depth_mask & workspace_mask)
            cloud = cloud[mask]
            seg = seg[mask]
            edge = edge[mask]
            # idxs = np.arange(len(cloud))
            idxs = np.random.choice(len(cloud), num_points, replace=True)
            cloud = cloud[idxs]
            seg = seg[idxs]
            edge = edge[idxs]
            if graspness_path is not None:
                graspness = np.load(project_path('data', graspness_path, scene, camera, str(view).zfill(4) + '.npy'))
                graspness = graspness.reshape(-1)
                graspness = graspness[idxs]

            result = []
            for i, idx in enumerate(np.unique(seg)):
                result += self.pc_plotly(torch.from_numpy(cloud[seg == idx]), size=1, color=px.colors.qualitative.Dark24[i])
            if with_pc:
                list = [cloud, seg[:, None]]
                if graspness_path is not None:
                    list.append(graspness[:, None])
                list.append(edge[:, None])
                if with_extrinsics:
                    return result, torch.from_numpy(np.concatenate(list, axis=-1)).float(), trans
                else:
                    return result, torch.from_numpy(np.concatenate(list, axis=-1)).float()
            return result, None
        else:
            raise ValueError('mode should be either pc or model')
    
    def get_scene_view_camera(
        self,
        scene: str,
        view: str,
        camera: str = 'kinect',
    ):
        path = project_path('data', 'DexGraspNet2.0', 'scenes', scene, camera)
        camera_pose = np.load(os.path.join(path, 'camera_poses.npy'))[int(view)]
        align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
        pose = np.linalg.inv(np.matmul(align_mat, camera_pose))
        x = -0.15
        y = -0.25
        z = 1.5
        eye = np.array([x, y, z, 1])
        center = np.array([x, y, 0, 1])
        eye = np.dot(pose, eye)[:3]
        center = np.dot(pose, center)[:3]
        names = ('x', 'y', 'z')
        return dict(
            up={k: v for k, v in zip(names, -pose[:3, 1])},
            center={k: v for k, v in zip(names, center)},
            eye={k: v for k, v in zip(names, eye)},
        )

    def show(self, 
             plotly_list: list,
             path: Optional[str] = None,
             scene: Optional[str] = None,
             view: Optional[str] = None,
             camera: Optional[str] = 'realsense',
    ) -> None:
        fig = go.Figure(data=plotly_list, layout=go.Layout(scene=dict(aspectmode='data')))
        if scene is not None and view is not None:
            camera = self.get_scene_view_camera(scene, view, camera)
            fig.update_layout(scene_camera=camera)
        if path is None:
            fig.show()
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.write_html(path)

def traj_plotly(vis_hand: Vis, vis_arm: Vis, ik: IK, traj: list):
    """plot trajectory of ur and hand
    traj: list of dict of arm->(6,) and hand->(joint_name->(1,)), both np.ndarray
    """
    result = []
    for action in traj:
        robot_color = random.choice(px.colors.sequential.Plasma)
        arm_plotly = vis_arm.robot_plotly(qpos=torch.from_numpy(action['arm'][None]).float(), opacity=0.5, color=robot_color, mesh_type='collision')
        rot, trans = ik.fk(action['arm'])
        hand_plotly = vis_hand.robot_plotly(trans=torch.from_numpy(trans)[None].float(), rot=torch.from_numpy(rot)[None].float(), qpos={k: torch.from_numpy(v)[None].float() for k, v in action['hand'].items()}, opacity=0.5, color=robot_color)
        result += arm_plotly + hand_plotly
    return result