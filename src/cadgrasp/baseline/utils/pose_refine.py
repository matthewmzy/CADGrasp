import os
import numpy as np
import torch
import trimesh
import trimesh.sample
from typing import List, Tuple

from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.baseline.utils.robot_info import GRIPPER_CENTER_SHIFT, GRIPPER_NEW_DEPTH, GRIPPER_SPACE_LEFT, GRIPPER_DEPTH_BASE, GRIPPER_HEIGHT, GRIPPER_MAX_WIDTH, GRIPPER_FINGER_WIDTH

class PoseRefine:
    """
        Refine gripper pose by closing it and fix depth
    """
    def __init__(self, N=10000):
        mesh_path = os.path.join('data', 'meshdata')
        self.obj_ids = [id for id in os.listdir(mesh_path) if id.isnumeric()]
        self.meshes = dict()
        self.fps = dict()
        self.vis = Vis('gripper')
        for id in self.obj_ids:
            self.meshes[id] = trimesh.load(os.path.join(mesh_path, id, 'nontextured_simplified.ply'))
        
        if not os.path.exists(os.path.join('data', 'fps')):
            os.makedirs(os.path.join('data', 'fps'))
            for id in self.obj_ids:
                samples = trimesh.sample.sample_surface_even(self.meshes[id], N)[0]
                if len(samples) < N:
                    idxs = np.random.randint(0, len(samples), N - len(samples))
                    samples = np.concatenate([samples, samples[idxs]])
                np.save(os.path.join('data', 'fps', id + '.npy'), samples)

        for id in self.obj_ids:
            self.fps[id] = torch.from_numpy(np.load(os.path.join('data', 'fps', id + '.npy'))).float()
    
    def find_intersect(self, obj_id: str, line_start: np.ndarray, line_end: np.ndarray):
        """
        obj_id: object id
        line_start, line_end: np.ndarray, (3, )
        return new line_start, line_end: np.ndarray, (3, )
        """

        mesh = self.meshes[obj_id]

        # Compute the direction vector of the line
        direction = line_end - line_start
        
        # Compute intersection points between the line segment and the mesh
        intersections, _, _ = mesh.ray.intersects_location(
            ray_origins=line_start.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=False,
        )
        
        return intersections
    
    def mask_pc(self, pc: torch.tensor, eps: float=0.0025):
        mask1 = (pc[:, 2] < GRIPPER_HEIGHT / 2 + eps) & (pc[:, 2] > -GRIPPER_HEIGHT / 2 - eps)
        mask2 = pc[:, 0] < GRIPPER_NEW_DEPTH + eps
        mask3 = (pc[:, 1] < GRIPPER_MAX_WIDTH) & (pc[:, 1] > -GRIPPER_MAX_WIDTH)
        return pc[mask1 & mask2 & mask3]
    
    def refine(self, 
               rot: torch.tensor, # (3, 3)
               trans: torch.tensor, # (3, )
               width: float,
               depth: float, 
               obj_id: int,
               obj_pose: torch.tensor, # (4, 4)
               table: torch.tensor = None, # (4,)
               extra: List[Tuple[int, torch.tensor]] = [], # [(obj_id, obj_pose)]
               eps: float = 0.0025,
    ):
        """
            Refine gripper pose by closing it and fix depth
            return rot, trans, width, depth
        """
        rot, trans, table = rot.float(), trans.float(), table.float()
        trans = trans - rot[:, 0] * (GRIPPER_NEW_DEPTH - depth)
        depth = GRIPPER_NEW_DEPTH
        obj_frame_rot = torch.einsum('ba,bc->ac', obj_pose[:3, :3], rot)
        obj_frame_trans = torch.einsum('ba,b->a', obj_pose[:3, :3], trans - obj_pose[:3, 3])

        obj_pc = torch.einsum('ba,nb->na', obj_frame_rot.float(), self.fps[str(obj_id).zfill(3)] - obj_frame_trans.float())
        extra = [(idx, pose) for idx, pose in extra if idx != obj_id]
        other_pcs = torch.stack([self.fps[str(idx).zfill(3)] for idx, _ in extra])
        other_poses = torch.stack([pose for _, pose in extra])
        other_frame_rots = torch.einsum('kba,bc->kac', other_poses[:, :3, :3], rot)
        other_frame_trans = torch.einsum('kba,kb->ka', other_poses[:, :3, :3], trans - other_poses[:, :3, 3])
        other_pc = torch.einsum('kba,knb->kna', other_frame_rots.float(), other_pcs - other_frame_trans.float()[:, None]).reshape(-1, 3)

        noise_trans = (torch.rand(5000, 3)*2-1) * 0.06 + trans
        height = (noise_trans * table[:3]).sum(-1) + table[3]
        table_pc = torch.einsum('ba,kb->ka', rot, noise_trans - height[:, None] * table[:3] - trans)
        other_pc = torch.cat([other_pc, table_pc], dim=0)

        obj_pc = self.mask_pc(obj_pc, eps=eps)
        other_pc = self.mask_pc(other_pc, eps=eps)

        close_bound_left = obj_pc[obj_pc[:, 1] < width/2 + 0.02, 1].max()
        close_bound_right = obj_pc[obj_pc[:, 1] > -width/2 - 0.02, 1].min()
        open_bound_left = 100
        open_bound_right = -100

        other_pcs_left = other_pc[other_pc[:, 1] > 0]
        other_pcs_right = other_pc[other_pc[:, 1] < 0]
        if len(other_pcs_left) > 0:
            open_bound_left = other_pcs_left[:, 1].min() 
        if len(other_pcs_right) > 0:
            open_bound_right = other_pcs_right[:, 1].max()
        
        space_left = max(0, open_bound_left - close_bound_left)
        space_right = max(0, close_bound_right - open_bound_right)
        # if min(space_left, space_right) == 0:
            # print('left', space_left, 'right', space_right)
            # print("!")
        assert min(space_left, space_right) > GRIPPER_FINGER_WIDTH + eps*2, f'left: {space_left}, right: {space_right}'

        new_space_left = min(0.01, (space_left - GRIPPER_FINGER_WIDTH) / 2)
        close_bound_left += new_space_left
        open_bound_left = close_bound_left + min(0.01, space_left - new_space_left * 2 - GRIPPER_FINGER_WIDTH)

        new_space_right = min(0.01, (space_right - GRIPPER_FINGER_WIDTH) / 2)
        close_bound_right -= new_space_right
        open_bound_right = close_bound_right - min(0.01, space_right - new_space_right * 2 - GRIPPER_FINGER_WIDTH)

        left_dist = np.random.rand() * (close_bound_left - open_bound_left) + open_bound_left
        right_dist = np.random.rand() * (open_bound_right - close_bound_right) + close_bound_right

        gripper_frame_new_center = torch.tensor([0, (left_dist + right_dist) / 2, 0]).float()
        trans = torch.einsum('ab,b->a', obj_pose[:3, :3], torch.einsum('ab,b->a', obj_frame_rot, gripper_frame_new_center) + obj_frame_trans) + obj_pose[:3, 3]
        
        # gripper_base = torch.tensor([-GRIPPER_DEPTH_BASE, (left_dist + right_dist) / 2, 0]).float()
        # gripper_frame_grasp_point = obj_pc[(obj_pc - gripper_base).norm(dim=-1).argmin()]
        # grasp_point = torch.einsum('ab,b->a', obj_pose[:3, :3], torch.einsum('ab,b->a', obj_frame_rot, gripper_frame_grasp_point) + obj_frame_trans) + obj_pose[:3, 3]

        obj_frame_center = torch.einsum('ab,b->a', obj_frame_rot, gripper_frame_new_center - torch.tensor([GRIPPER_DEPTH_BASE, 0, 0]).float()) + obj_frame_trans
        obj_frame_dir = torch.einsum('ab,b->a', obj_frame_rot, gripper_frame_new_center + torch.tensor([GRIPPER_DEPTH_BASE, 0, 0]).float()) + obj_frame_trans
        grasp_point = torch.from_numpy(self.find_intersect(str(obj_id).zfill(3), obj_frame_center, obj_frame_dir)[0]).float()
        assert torch.norm(obj_frame_center - grasp_point) < GRIPPER_DEPTH_BASE + GRIPPER_NEW_DEPTH
        grasp_point = torch.einsum('ab,b->a', obj_pose[:3, :3], grasp_point) + obj_pose[:3, 3]

        return rot, trans, left_dist - right_dist, GRIPPER_NEW_DEPTH, grasp_point
