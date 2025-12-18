import os, sys
from typing import Union
from collections import OrderedDict
import numpy as np
import transforms3d
import trimesh as tm
import torch
import yaml
import random
import plotly.express as px

from urdf_parser_py.urdf import Robot, Box, Mesh, Cylinder
from torchprimitivesdf import box_sdf, transform_points_inverse, fixed_transform_points_inverse
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go


class RobotPoints:
    """
    class to store robot points
    """
    
    def __init__(
        self, 
        points_dict: OrderedDict, 
    ):
        """
        Args:
        - points_dict: OrderedDict, points of each link
        """
        # data
        # - _points_dict: OrderedDict, link name -> points, (n_points, 3)
        # - _points: torch.Tensor, concatenated points, (n_points, 3)
        self._points_dict: dict[str, torch.Tensor] = points_dict
        if any([type(points_dict[link_name]) != torch.Tensor for link_name in points_dict]):
            for link_name in points_dict:
                points_dict[link_name] = torch.tensor(points_dict[link_name], 
                    dtype=torch.float32)
        self._points: torch.Tensor = torch.cat(
            [points_dict[link_name] for link_name in points_dict], dim=0)
        
        # indexing
        # - _global_index_to_link_index: torch.Tensor, index in _points -> index in _points_dict
        # - points_link_indices: torch.Tensor, link index for each point
        self._global_index_to_link_index = sum([[i] * len(points_dict[link_name]) 
            for i, link_name in enumerate(points_dict)], [])
        self._global_index_to_link_index: torch.Tensor = torch.tensor(
            self._global_index_to_link_index, dtype=torch.long, device=self.device)
        self.points_link_indices = [
            link_index * torch.ones(len(self._points_dict[link_name]), dtype=torch.long) 
            for link_index, link_name in enumerate(self._points_dict)
        ]
        self.points_link_indices: torch.Tensor = torch.cat(self.points_link_indices)
    
    @property
    def device(self):
        """
        torch.device: device
        """
        return self._points.device

    @property
    def n_points(self):
        """
        int: number of points
        """
        return len(self._points)
    
    def to(self, device: torch.device):
        """
        move to device
        
        Args:
        - device: torch.device, device
        """
        points_dict = OrderedDict()
        for link_name in self._points_dict:
            points_dict[link_name] = self._points_dict[link_name].to(device)
        robot_points = RobotPoints(points_dict)
        return robot_points
    
    def get_link_points(
        self, 
        link_name: str,
    ):
        """
        get points of a link
        """
        return self._points_dict[link_name]
    
    def get_points(
        self, 
        local_translations: dict,
        local_rotations: dict, 
        global_translation: Union[torch.Tensor, None] = None,
        global_rotation: Union[torch.Tensor, None] = None,
        robot_frame: bool = False,
        link_names: list = None,
    ):
        """
        get points in global frame
        
        Args:
        - local_translations: dict, link name -> translation, (batch_size, 3)
        - local_rotations: dict, link name -> rotation, (batch_size, 3, 3)
        - global_translation: torch.Tensor, global translation, (batch_size, 3)
        - global_rotation: torch.Tensor, global rotation, (batch_size, 3, 3)
        - robot_frame: bool, whether to return points in robot frame
        - link_names: list, link names to return, if None, return all
        
        Returns:
        - points: torch.Tensor, (batch_size, n_points, 3)
        """
        if link_names is None:
            link_names = self._points_dict.keys()
        points = []
        for link_name in link_names:
            if len(self._points_dict[link_name]) == 0:
                continue
            local_translation = local_translations[link_name]
            local_rotation = local_rotations[link_name]
            link_points = self._points_dict[link_name].to(local_translation.device)
            points.append(link_points @ local_rotation.transpose(1, 2) + \
                local_translation.unsqueeze(1))
        points = torch.cat(points, dim=1)
        if not robot_frame:
            points = points @ global_rotation.transpose(1, 2) + global_translation.unsqueeze(1)
        return points


class RobotModel:
    """
    class to load robot model from urdf and compute forward kinematics
    """
    
    def __init__(
        self, 
        urdf_path: str,
        meta_path: str,
    ):
        """
        Args:
        - urdf_path: str, path to the urdf file
        - meta_path: str, path to the meta file
        """
        
        # load urdf
        self._urdf_path: str = urdf_path
        self._robot = Robot.from_xml_file(urdf_path)
        
        # load meta
        self._meta = yaml.safe_load(open(meta_path, 'r'))
        
        # compute joint order
        self._compute_joint_order()
        
        # build articulation
        self._build_articulation()

        # build geometry
        self._build_geometry()
        
        # build collision mask
        self._build_collision_mask()

        self.surface_points = None

        # Define link names for thumb and other fingers (to be filled from URDF)
        self.thumb_link_names = ['thumb_dip', 'thumb_fingertip', 'thumb_pip']  # TODO: fill with actual thumb link names
        self.other_finger_link_names = ['pip', 'pip_2', 'pip_3','dip', 'dip_2', 'dip_3', 'fingertip', 'fingertip_2', 'fingertip_3']  # TODO: fill with actual other finger link names
    
    def _compute_joint_order(self):
        """
        get joint order, use the dfs order, break ties by insertion order. \n
        joint_names: list, the order for parsing and forward kinematics
        """
        # get root link
        link_names = [link.name for link in self._robot.links]
        child_link_names = [joint.child for joint in self._robot.joints]
        self._root_link_name: str = list(set(link_names) - set(child_link_names))[0]
        # tree search
        self.joint_names: list[str] = []
        stack = list(reversed([joint for joint in self._robot.joints 
            if joint.parent == self._root_link_name]))
        while stack:
            # get the last item
            joint = stack[-1]
            self.joint_names.append(joint.name)
            stack = stack[:-1]
            # add children to the priority queue
            children = [child for child in self._robot.joints if child.parent == joint.child]
            stack += list(reversed(children))
    
    def _build_articulation(self):
        """
        build articulation from urdf. 
        parse joint tree, axis cross product matrix, origin transform
        """
        self._joints_parent: list[str] = []
        self._joints_child: list[str] = []
        self._joints_type: list[str] = []
        self.joints_upper: list[float] = []
        self.joints_lower: list[float] = []
        self._joints_axis_K: list[np.ndarray] = []          # cross product matrix, 3x3
        self._joints_rotation: list[np.ndarray] = []        # rotation matrix, 3x3
        self._joints_translation: list[np.ndarray] = []     # translation, 3
        for joint_name in self.joint_names:
            joint = [joint for joint in self._robot.joints if joint.name == joint_name][0]
            # joint information
            self._joints_parent.append(joint.parent)
            self._joints_child.append(joint.child)
            self._joints_type.append(joint.type)
            self.joints_upper.append(joint.limit.upper)
            self.joints_lower.append(joint.limit.lower)
            # joint axis cross product matrix 
            if joint.type == 'revolute':
                eye = np.eye(3, dtype=np.float32)
                axis = np.array(joint.axis, dtype=np.float32)
                self._joints_axis_K.append(np.cross(eye, axis))
            else:
                self._joints_axis_K.append(None)
            # joint origin transform
            translation, rotation_matrix, _ = self._get_frame_transform(joint.origin)
            self._joints_translation.append(translation)
            self._joints_rotation.append(rotation_matrix)
        self.n_dofs: int = len([joint_type for joint_type in self._joints_type 
            if joint_type != 'fixed'])
    
    def _build_geometry(self):
        """
        build geometry from urdf.
        load collision and visual meshes. 
        """
        self._geometry: OrderedDict[str, dict] = OrderedDict()
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(self._urdf_path)))
        for link in sorted(self._robot.links, key=lambda link: link.name):
            if link.visual is None and link.collision is None:
                continue
            link_name = link.name
            self._geometry[link_name] = {}
            # load collision meshes
            collision_mesh = tm.Trimesh()
            self._geometry[link_name]['boxes'] = []
            self._geometry[link_name]['capsules'] = []
            for collision in link.collisions:
                if type(collision.geometry) == Box:
                    translation, rotation, transform = self._get_frame_transform(collision.origin)
                    size = np.array(collision.geometry.size, dtype=np.float32)
                    box_mesh = tm.primitives.Box(size, transform)
                    collision_mesh += box_mesh
                    self._geometry[link_name]['boxes'].append({
                        'translation': translation, 
                        'rotation': rotation, 
                        'size': size / 2, 
                    })
                elif type(collision.geometry) == Cylinder:
                    # interpret cylinder as capsule since urdf-parser does not support capsule
                    translation, rotation, transform = self._get_frame_transform(collision.origin)
                    radius = collision.geometry.radius
                    height = collision.geometry.length
                    capsule_mesh = tm.primitives.Capsule(radius, height, transform)
                    collision_mesh += capsule_mesh
                    self._geometry[link_name]['capsules'].append({
                        'translation': translation, 
                        'rotation': rotation, 
                        'radius': radius, 
                        'height': height, 
                    })
                elif type(collision.geometry) == Mesh:
                    filename = os.path.join(base_path, collision.geometry.filename[10:])
                    collision_mesh += tm.load_mesh(filename)
                else:
                    raise ValueError(f'Unsupported geometry type: {type(collision.geometry)}')
            self._geometry[link_name].update({
                'collision_vertices': collision_mesh.vertices,
                'collision_faces': collision_mesh.faces,
            })
            # load visual mesh
            visual = link.visual
            scale = getattr(visual.geometry, 'scale', [1, 1, 1])
            scale = [1, 1, 1] if scale is None else scale
            transform = self._get_frame_transform(visual.origin)[2]
            filename = os.path.join(base_path, visual.geometry.filename[10:])
            visual_mesh = tm.load(
                filename, force='mesh').apply_scale(scale).apply_transform(transform)
            self._geometry[link_name].update({
                    'visual_vertices': visual_mesh.vertices,
                    'visual_faces': visual_mesh.faces,
                })
    
    def _build_collision_mask(self):
        """
        build collision mask from urdf. 
        """
        self.adjacency_mask = torch.eye(len(self.link_names), dtype=torch.bool)
        for joint in self._robot.joints:
            if joint.parent in self.link_names and joint.child in self.link_names:
                parent_id = self.link_names.index(joint.parent)
                child_id = self.link_names.index(joint.child)
                self.adjacency_mask[parent_id, child_id] = True
                self.adjacency_mask[child_id, parent_id] = True
        for pair in self._meta['adjacency_pairs']:
            if pair[0] in self.link_names and pair[1] in self.link_names:
                id_0 = self.link_names.index(pair[0])
                id_1 = self.link_names.index(pair[1])
                self.adjacency_mask[id_0, id_1] = True
                self.adjacency_mask[id_1, id_0] = True
    
    @staticmethod
    def _get_frame_transform(frame):
        """
        extract translation, rotation matrix, and transform from frame
        """
        translation = getattr(frame, 'xyz', [0, 0, 0])
        translation = np.array(translation, dtype=np.float32)
        rotation_euler = getattr(frame, 'rpy', [0, 0, 0])
        rotation_matrix = transforms3d.euler.euler2mat(*rotation_euler, 'sxyz')
        rotation_matrix = rotation_matrix.astype(np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return translation, rotation_matrix, transform
    
    def get_default_joint_indices(self):
        """
        get default joint indices, use the order in joint_names, 
        ignore fixed joints
        Returns:
        - joint_indices: dict, joint name -> qpos index
        """
        joint_order_movable = [joint_name for joint_name in self.joint_names
            if self._joints_type[self.joint_names.index(joint_name)] != 'fixed']
        joint_indices = {joint_name: i for i, joint_name in enumerate(joint_order_movable)}
        return joint_indices
    
    def forward_kinematics(
        self, 
        qpos_dict: dict,
    ):
        """
        compute forward kinematics
        
        Args:
        - qpos_dict: dict[str, torch.Tensor[batch_size]], joint_name -> qpos
        
        Returns:
        - link_translations: dict[str, torch.Tensor[batch_size, 3]], link name -> translation
        - link_rotations: dict[str, torch.Tensor[batch_size, 3, 3]], link name -> rotation
        """
        
        device = qpos_dict[self.joint_names[0]].device
        batch_size = qpos_dict[self.joint_names[0]].shape[0]

        assert all([qpos_dict[joint_name].shape[0] == batch_size for joint_name in qpos_dict])
        assert all([qpos_dict[joint_name].device == device for joint_name in qpos_dict])
        
        # forward kinematics
        
        link_translations = {}
        link_rotations = {}
        link_translations[self._root_link_name] = torch.zeros([batch_size, 3], 
            dtype=torch.float32, device=device)
        link_rotations[self._root_link_name] = torch.eye(3, 
            dtype=torch.float32, device=device).expand(batch_size, 3, 3).contiguous()
        
        for joint_num, joint_name in enumerate(self.joint_names):
            child_name = self._joints_child[joint_num]
            parent_name = self._joints_parent[joint_num]
            joint_type = self._joints_type[joint_num]
            # get parent transform
            parent_translation = link_translations[parent_name]
            parent_rotation = link_rotations[parent_name]
            # compute joint transform
            joint_translation = torch.from_numpy(self._joints_translation[joint_num]).to(device)
            joint_rotation = torch.from_numpy(self._joints_rotation[joint_num]).to(device)
            if joint_type == 'revolute':
                angle = qpos_dict[joint_name].reshape(-1, 1, 1)
                K = torch.from_numpy(self._joints_axis_K[joint_num]).to(device)
                eye = torch.eye(3, dtype=torch.float32, device=device)
                # Rodrigues' formula
                axis_rotation = eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
                joint_rotation = joint_rotation @ axis_rotation
            # compute child transform
            child_translation = parent_rotation @ joint_translation + parent_translation
            child_rotation = parent_rotation @ joint_rotation
            link_translations[child_name] = child_translation
            link_rotations[child_name] = child_rotation
        
        return link_translations, link_rotations

    def clamp_qpos(
        self, 
        qpos_dict: dict,
    ):
        """
        clamp qpos to joint limits, inplace
        
        Args:
        - qpos_dict: dict[str, torch.Tensor[batch_size]], joint_name -> qpos
        """
        for joint_index, joint_name in enumerate(self.joint_names):
            lower = self.joints_lower[joint_index]
            upper = self.joints_upper[joint_index]
            qpos_dict[joint_name][:] = torch.clamp(qpos_dict[joint_name], lower, upper)
    
    @property
    def link_names(self):
        """
        get link names
        """
        return list(self._geometry.keys())

    def get_ft_center(self, link_translations: dict, link_rotations: dict, link_name: str, y_bias: float = 0.0):
        """
        get link center
        
        Args:
        - link_translations: dict[str, torch.Tensor[batch_size, 3]], link name -> translation
        - link_rotations: dict[str, torch.Tensor[batch_size, 3, 3]], link name -> rotation
        - link_name: str, link name
        - y_bias: float, result = y - y_bias
        
        Returns:
        - link_center: torch.Tensor[batch_size, 3], link name -> center
        """
        device = link_translations[link_name].device
        boxes = self.get_link_mesh(link_name, 'collision')[0].reshape(2, -1, 3).mean(1).to(device).float()
        boxes = boxes[boxes[:,1].argmin()]
        boxes[..., 1] -= y_bias
        result = torch.einsum('nab,b->na', link_rotations[link_name], boxes) + link_translations[link_name]
        return result
    
    def get_link_mesh(self, link_name: str, mesh_type: str):
        """
        get link mesh
        
        Args:
        - link_name: str, link name
        - mesh_type: str, 'collision' or 'visual'
        
        Returns:
        - vertices: torch.Tensor, vertices, [n_vertices, 3]
        - faces: torch.Tensor, faces, [n_faces, 3]
        """
        vertices = torch.from_numpy(self._geometry[link_name][f'{mesh_type}_vertices'])
        faces = torch.from_numpy(self._geometry[link_name][f'{mesh_type}_faces'])
        return vertices, faces

    
    def sample_surface_points_full(
        self,
        qpos_dict: dict,
        n_points: int = None,
        n_points_each_link: int = None,
    ):
        """
        compute forward kinematics and sample surface points from robot surface meshe
        
        Args:
        - qpos_dict: dict[str, torch.Tensor[batch_size]], joint_name -> qpos
        - n_points: int, number of points to sample in total
        - n_points_each_link: Union[int, None], number of points to sample for each link
        
        Returns:
        - points: dict[str, torch.Tensor], sampled points
        """
        if self.surface_points is None:
            self.surface_points = self.sample_surface_points(n_points=n_points, n_points_each_link=n_points_each_link)
        link_trans, link_rot = self.forward_kinematics(qpos_dict)
        return torch.cat([torch.einsum('nab,kb->nka', link_rot[k], self.surface_points[k]) + link_trans[k][:, None] for k in link_trans.keys()], dim=1)

    def sample_surface_points(
        self, 
        n_points: int = None,
        n_points_each_link: int = None,
    ):
        """
        sample surface points from robot surface meshe
        
        Args: 
        - n_points: int, number of points to sample in total
        - n_points_each_link: Union[int, None], number of points to sample for each link
        Returns:
        - points: dict[str, torch.Tensor], sampled points
        """
        assert n_points_each_link is None or n_points is None
        if n_points_each_link is not None:
            num_samples = dict([
                (link_name, n_points_each_link 
                if len(self._geometry[link_name]['collision_vertices']) > 0 else 0) 
                for link_name in self._geometry ])
        else:
            # compute areas
            areas = {}
            for link_name in self._geometry:
                link_mesh = tm.Trimesh(
                    vertices=self._geometry[link_name]['collision_vertices'], 
                    faces=self._geometry[link_name]['collision_faces']
                )
                areas[link_name] = link_mesh.area
            total_area = sum(areas.values())
            # compute number of samples for each link
            num_samples = dict([(link_name, int(areas[link_name] / total_area * n_points)) 
                for link_name in self._geometry])
            num_samples[list(num_samples.keys())[0]] += n_points - sum(num_samples.values())
        # sample points
        points = {}
        for link_name in self._geometry:
            if num_samples[link_name] == 0:
                points[link_name] = torch.tensor([], dtype=torch.float).reshape(0, 3)
                continue
            vertices = torch.tensor(self._geometry[link_name]['collision_vertices'], dtype=torch.float)
            faces = torch.tensor(self._geometry[link_name]['collision_faces'], dtype=torch.long)
            mesh = pytorch3d.structures.Meshes(
                vertices.unsqueeze(0), 
                faces.unsqueeze(0)
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=10 * num_samples[link_name])
            num_collision_vertices = len(self._geometry[link_name]['collision_vertices'])
            if num_collision_vertices <= 16:     # collision mesh is one or two boxes
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, 
                    K=num_samples[link_name] - len(self._geometry[link_name]['collision_vertices']))[0][0]
                points[link_name] = torch.cat([
                    surface_points, 
                    vertices, 
                ])
            else:
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, 
                    K=num_samples[link_name])[0][0]
                points[link_name] = surface_points
        return points
    
    def get_finger_surface_points(
        self, 
        translation: torch.Tensor,
        rotation: torch.Tensor,
        qpos_dict: dict, 
        n_points: int = None, 
        n_points_each_link: int = None
    ):
        """
        Get the surface points of the thumb and other fingers based on the given qpos_dict.
        
        Args:
        - qpos_dict: dict[str, torch.Tensor[batch_size]], joint_name -> qpos
        - n_points: int, total number of points to sample across all links
        - n_points_each_link: int, number of points to sample for each link
        
        Returns:
        - hand_thumb_points: torch.Tensor[batch_size, n_points_thumb, 3], surface points of the thumb
        - hand_other_finger_points: torch.Tensor[batch_size, n_points_other, 3], surface points of the other fingers
        """
        batch_size = qpos_dict[self.joint_names[0]].shape[0]
        trans = translation if translation is not None else torch.zeros(batch_size, 3)
        rot = rotation if rotation is not None else torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

        # Ensure surface points are sampled if not already done
        if self.surface_points is None:
            self.surface_points = self.sample_surface_points(n_points=n_points, n_points_each_link=n_points_each_link)
        
        # Create a RobotPoints instance with the sampled surface points
        robot_points = RobotPoints(self.surface_points)
        
        # Compute forward kinematics to get link transformations
        link_translations, link_rotations = self.forward_kinematics(qpos_dict)
        link_translations = {k: torch.einsum('nab,nb->na', rot, v) + trans for k, v in link_translations.items()}
        link_rotations = {k: torch.einsum('nab,nbc->nac', rot, v) for k, v in link_rotations.items()}
        
        # Get thumb surface points
        hand_thumb_points = robot_points.get_points(
            local_translations=link_translations,
            local_rotations=link_rotations,
            robot_frame=True,
            link_names=self.thumb_link_names
        )
        
        # Get other fingers' surface points
        hand_other_finger_points = robot_points.get_points(
            local_translations=link_translations,
            local_rotations=link_rotations,
            robot_frame=True,
            link_names=self.other_finger_link_names
        )

        hand_whole_points = robot_points.get_points(
            local_translations=link_translations,
            local_rotations=link_rotations,
            robot_frame=True,
            link_names=self.link_names
        )
        
        return hand_thumb_points, hand_other_finger_points, hand_whole_points
    
    def cal_distance(
        self, 
        local_translations: dict,
        local_rotations: dict, 
        global_translation: torch.Tensor,
        global_rotation: torch.Tensor,
        x: torch.Tensor, 
        dilation_pen: float = 0.,
    ):
        """
        Calculate signed distances from x to robot surface meshes \n
        Interiors are positive, exteriors are negative
        
        Args: 
        - local_translations: dict, link name -> translation, (batch_size, 3)
        - local_rotations: dict, link name -> rotation, (batch_size, 3, 3)
        - global_translation: torch.Tensor, (batch_size, 3), global translation
        - global_rotation: torch.Tensor, (batch_size, 3, 3), global rotation
        - x: torch.Tensor, (batch_size, n_points, 3), batched point cloud
        - dilation_pen: float, penetration dilation
        
        Returns: 
        - distance: torch.Tensor, (batch_size, n_points), signed distance
        """
        dis = []
        x = transform_points_inverse(x, global_translation, global_rotation)
        for link_name in self._geometry:
            local_translation = local_translations[link_name]
            local_rotation = local_rotations[link_name]
            x_local = transform_points_inverse(x, local_translation, local_rotation)
            x_local = x_local.reshape(-1, 3)
            for box in self._geometry[link_name]['boxes']:
                box_translation = torch.tensor(box['translation'], 
                    dtype=torch.float32, device=x.device)
                box_rotation = torch.tensor(box['rotation'],
                    dtype=torch.float32, device=x.device)
                x_box = fixed_transform_points_inverse(x_local, box_translation, box_rotation)
                size = box['size'] + dilation_pen
                size = torch.tensor(size, dtype=torch.float32, device=x.device)
                dis_local, dis_signs, _ = box_sdf(x_box, size)
                dis_local = (dis_local + 1e-8).sqrt()
                dis_local = torch.where(dis_signs, -dis_local, dis_local)
                dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
            for capsule in self._geometry[link_name]['capsules']:
                capsule_translation = torch.tensor(capsule['translation'],
                    dtype=torch.float32, device=x.device)
                capsule_rotation = torch.tensor(capsule['rotation'],
                    dtype=torch.float32, device=x.device)
                x_capsule = fixed_transform_points_inverse(x_local, 
                    capsule_translation, capsule_rotation)
                radius = capsule['radius'] + dilation_pen
                height = capsule['height']
                nearest_point = x_capsule.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], -height / 2, height / 2)
                vec = x_capsule - nearest_point
                dis_local = radius - ((vec * vec).sum(-1) + 1e-8).sqrt()
                dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis
    
    def cal_self_distance(
        self, 
        robot_surface_points: RobotPoints,
        local_translations: dict,
        local_rotations: dict, 
        dilation_spen: float = 0., 
    ):
        """
        Calculate the distance of each surface point to the robot surface
        
        Args:
        - robot_surface_points: RobotPoints, robot surface points
        - local_translations: dict, link name -> translation, (batch_size, 3)
        - local_rotations: dict, link name -> rotation, (batch_size, 3, 3)
        - dilation_spen: float, self penetration dilation
        """
        device = local_translations[self.link_names[0]].device
        # move adjacency mask and point_link_indices to device
        adjacency_mask = self.adjacency_mask.to(device)
        points_link_indices = robot_surface_points.points_link_indices.to(device)
        # get surface points: (total_batch_size, n_surface_points, 3)
        x = robot_surface_points.get_points(
            local_translations=local_translations,
            local_rotations=local_rotations,
            robot_frame=True)
        # cal distance
        dis = []
        for link_index, link_name in enumerate(self._geometry):
            local_translation = local_translations[link_name]
            local_rotation = local_rotations[link_name]
            x_local = transform_points_inverse(x, local_translation, local_rotation)
            x_local = x_local.reshape(-1, 3)
            for box in self._geometry[link_name]['boxes']:
                box_translation = torch.tensor(box['translation'], 
                    dtype=torch.float32, device=x.device)
                box_rotation = torch.tensor(box['rotation'],
                    dtype=torch.float32, device=x.device)
                x_box = fixed_transform_points_inverse(x_local, box_translation, box_rotation)
                size = box['size'] + dilation_spen
                size = torch.tensor(size, dtype=torch.float32, device=x.device)
                dis_local, dis_signs, _ = box_sdf(x_box, size)
                dis_local = (dis_local + 1e-8).sqrt()
                dis_local = torch.where(dis_signs, -dis_local, dis_local)
                dis_local = dis_local.reshape(x.shape[0], x.shape[1])  # (total_batch_size, n_surface_points)
                is_adjacent = adjacency_mask[link_index, points_link_indices]  # (n_surface_points,)
                dis_local[:, is_adjacent] = -float('inf')
                dis.append(dis_local)
            for capsule in self._geometry[link_name]['capsules']:
                capsule_translation = torch.tensor(capsule['translation'],
                    dtype=torch.float32, device=x.device)
                capsule_rotation = torch.tensor(capsule['rotation'],
                    dtype=torch.float32, device=x.device)
                x_capsule = fixed_transform_points_inverse(x_local, 
                    capsule_translation, capsule_rotation)
                radius = capsule['radius'] + dilation_spen
                height = capsule['height']
                nearest_point = x_capsule.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], -height / 2, height / 2)
                vec = x_capsule - nearest_point
                dis_local = radius - ((vec * vec).sum(-1) + 1e-8).sqrt()
                dis_local = dis_local.reshape(x.shape[0], x.shape[1])  # (total_batch_size, n_surface_points)
                is_adjacent = adjacency_mask[link_index, points_link_indices]  # (n_surface_points,)
                dis_local[:, is_adjacent] = -float('inf')
                dis.append(dis_local)
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(
        self, 
        robot_surface_points: RobotPoints, 
        local_translations: dict,
        local_rotations: dict, 
        dilation_spen: float = 0.,
    ):
        """
        Calculate self penetration
        
        Args:
        - robot_surface_points: RobotPoints, robot surface points
        - local_translations: dict, link name -> translation, (batch_size, 3)
        - local_rotations: dict, link name -> rotation, (batch_size, 3, 3)
        - dilation_spen: float, self penetration dilation
        Returns:
        - spen: torch.Tensor, (batch_size), self penetration
        """
        dis = self.cal_self_distance(
            robot_surface_points=robot_surface_points,
            local_translations=local_translations,
            local_rotations=local_rotations,
            dilation_spen=dilation_spen, 
        )
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen
    
    def cal_dis_plane_each(
        self, 
        local_translations: dict,
        local_rotations: dict, 
        global_translation: torch.Tensor,
        global_rotation: torch.Tensor,
        p: torch.Tensor, 
        dilation_tpen: float = 0,
    ):
        """
        Calculate the signed distance from each link to the plane, 
        positive below the plane, negative above the plane
        
        Args:
        - local_translations: dict, link name -> translation, (batch_size, 3)
        - local_rotations: dict, link name -> rotation, (batch_size, 3, 3)
        - global_translation: torch.Tensor, (batch_size, 3), global translation
        - global_rotation: torch.Tensor, (batch_size, 3, 3), global rotation
        - p: torch.Tensor, (batch_size, 4), plane parameters, ax + by + cz + d >= 0 above
        - dilation_tpen: float, penetration dilation
        
        Returns:
        - dis: torch.Tensor, (batch_size, n_links), signed distance
        """
        dis = []
        p = p.clone()
        p[:, 3] += (p[:, :3] * global_translation).sum(-1)
        p[:, :3] = (p[:, :3].unsqueeze(1) @ global_rotation).squeeze(1)
        for link_name in self._geometry:
            local_translation = local_translations[link_name]
            local_rotation = local_rotations[link_name]
            p_local = p.clone()
            p_local[:, 3] += (p[:, :3] * local_translation).sum(-1)
            p_local[:, :3] = (p[:, :3].unsqueeze(1) @ local_rotation).squeeze(1)
            dis_locals = []
            for box in self._geometry[link_name]['boxes']:
                box_translation = torch.tensor(box['translation'], 
                    dtype=torch.float32, device=p.device)
                box_rotation = torch.tensor(box['rotation'],
                    dtype=torch.float32, device=p.device)
                p_box = p_local.clone()
                p_box[:, 3] += (p_local[:, :3] * box_translation).sum(-1)
                p_box[:, :3] = p_local[:, :3] @ box_rotation
                size = box['size'] + dilation_tpen
                size = torch.tensor(size, dtype=torch.float32, device=p.device)
                box_vertices = size * torch.tensor([
                    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], 
                    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
                ], dtype=torch.float32, device=p.device)
                dis_local = (p_box[:, :3] @ box_vertices.T + p_box[:, 3:]).min(-1)[0]
                dis_locals.append(dis_local)
            for capsule in self._geometry[link_name]['capsules']:
                capsule_translation = torch.tensor(capsule['translation'],
                    dtype=torch.float32, device=p.device)
                capsule_rotation = torch.tensor(capsule['rotation'],
                    dtype=torch.float32, device=p.device)
                p_capsule = p_local.clone()
                p_capsule[:, 3] += (p_local[:, :3] * capsule_translation).sum(-1)
                p_capsule[:, :3] = p_local[:, :3] @ capsule_rotation
                radius = capsule['radius'] + dilation_tpen
                height = capsule['height']
                capsule_points = torch.tensor([
                    [0, 0, -height / 2], [0, 0, height / 2]
                ], dtype=torch.float32, device=p.device)
                dis_local = (p_capsule[:, :3] @ capsule_points.T + p_capsule[:, 3:]).min(-1)[0]
                dis_local -= radius
                dis_locals.append(dis_local)
            dis_local = torch.stack(dis_locals, dim=1).min(-1)[0]
            dis.append(-dis_local)
        dis = torch.stack(dis, dim=1)
        return dis
    
    def cal_dis_plane(
        self, 
        local_translations: dict,
        local_rotations: dict,
        global_translation: torch.Tensor,
        global_rotation: torch.Tensor,
        p: torch.Tensor,
        dilation_tpen: float = 0,
    ):
        """
        Calculate the signed distance from the robot to the plane, 
        positive below the plane, negative above the plane
        
        Args:
        - local_translations: dict, link name -> translation, (batch_size, 3)
        - local_rotations: dict, link name -> rotation, (batch_size, 3, 3)
        - global_translation: torch.Tensor, (batch_size, 3), global translation
        - global_rotation: torch.Tensor, (batch_size, 3, 3), global rotation
        - p: torch.Tensor, (batch_size, 4), plane parameters, ax + by + cz + d >= 0 above
        - dilation_tpen: float, penetration dilation
        
        Returns:
        - dis: torch.Tensor, (batch_size), signed distance
        """
        dis = self.cal_dis_plane_each(
            local_translations=local_translations,
            local_rotations=local_rotations,
            global_translation=global_translation,
            global_rotation=global_rotation,
            p=p,
            dilation_tpen=dilation_tpen,
        )
        dis = dis.max(-1)[0]
        return dis

if __name__ == '__main__':
    robot_model = RobotModel(
        urdf_path='robot_models/urdf/leap_hand_simplified.urdf',
        meta_path='robot_models/meta/leap_hand/meta.yaml'
    )

    print(f'joint names: {robot_model.joint_names}')
    print(f'link names: {robot_model.link_names}')
    
    # Set up qpos_dict with zeros for movable joints
    qpos_dict = {joint_name: torch.tensor([random.uniform(l,u)]) for joint_name, l, u in zip(robot_model.joint_names, robot_model.joints_lower, robot_model.joints_upper)}
    trans = torch.tensor([[1,2,3]],dtype=torch.float)
    rot = torch.tensor([[[1,0,0],[0,0,-1],[0,1,0]]],dtype=torch.float)
    # Get finger surface points
    hand_thumb_points, hand_other_finger_points, hand_whole_points = \
        robot_model.get_finger_surface_points(trans, rot, qpos_dict, n_points_each_link=1000)

    sys.path.append('.')
    from src.utils.vis_plotly import Vis
    vis = Vis(
        robot_name='leap_hand',
        urdf_path='robot_models/urdf/leap_hand_simplified.urdf',
        meta_path='robot_models/meta/leap_hand/meta.yaml'
    )
    hand_mesh_plotly = vis.robot_plotly(
        trans,
        rot,
        qpos_dict, 
        opacity=0.5, 
        color="red")
    
    # Prepare data for plotting
    thumb_points = hand_thumb_points[0]  # shape (n_points_thumb, 3)
    other_points = hand_other_finger_points[0]  # shape (n_points_other, 3)
    
    thumb_plotly = vis.pc_plotly(thumb_points, size=1, color="blue")
    other_plotly = vis.pc_plotly(other_points, size=1, color="green")
    
    vis.show(hand_mesh_plotly+thumb_plotly+other_plotly)