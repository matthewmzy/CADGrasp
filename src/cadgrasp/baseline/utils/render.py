import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh as tm
import numpy as np
import torch
import torch.nn.functional
from tqdm import tqdm
import json

from cadgrasp.baseline.utils.observer import Observer

class Renderer:
    def __init__(self, image_width, image_height, intrinsics, gpu):
        self.device = torch.device(gpu)
        self.observer = Observer(image_width, image_height, intrinsics, gpu)
        for object_code in tqdm(os.listdir('data/meshdata'), desc='loading meshes'):
            if object_code.isdigit():
                object_mesh = tm.load(os.path.join('data/meshdata', object_code, 'nontextured.ply'), process=False).apply_scale(1.0)
                self.observer.add_object_mesh(object_code, object_mesh, int(object_code)+1) # table is 0
        
    def render(self, object_code_list, obj_transform, table_transform):
        num_objs = len(object_code_list)

        # render pcs
        object_transform = np.zeros([num_objs + 1, 4, 4], dtype=np.float32)
        for i in range(num_objs):
            object_transform[i, :3, :3] = obj_transform[i][:3, :3]
            object_transform[i, :3, 3] = obj_transform[i][:3, 3]
            object_transform[i, 3, 3] = 1

        object_code_list.append('table')
        object_transform[-1] = table_transform

        images = (torch.tensor(self.observer.render_point_cloud(object_code_list, object_transform), dtype=torch.float, device=self.device))
        return images
        # apply gaussian filter to images
        # if gaussian_filter:
            # half_size = torch.randint(0, 6, [1], dtype=torch.int, device=self.device).item()
            # images[:, :, :, 2] = torch.nn.functional.avg_pool2d(images[:, None, :, :, 2], 2 * half_size + 1, stride=1, padding=half_size, count_include_pad=False).squeeze(1)
        # images = images.reshape(num_envs, -1, 4)
        # transform to world frame
        # view = torch.tensor(self.observer.View.to_list(), dtype=torch.float, device=self.device).T
        # images[:, :, :3] = (images[:, :, :3] - view[:3, 3]) @ view[:3, :3]
        # convert to pcs
        # object_mask = (images[:, :, 3] == Id.OBJECT.value) & (images[:, :, :3] > self.pc_p1).all(dim=2) & (images[:, :, :3] < self.pc_p2).all(dim=2)
        # num_object_points = self.n_camera_pc # torch.minimum(torch.sum(object_mask, dim=1), torch.tensor(self.n_camera_pc, dtype=torch.int, device=self.device))
        # num_table_points = self.n_camera_pc + self.n_camera_pc_table - num_object_points
        # pcs = torch.zeros([num_envs, self.n_camera_pc, 3], dtype=torch.float, device=self.device)
        # invalid = torch.zeros([num_envs,], dtype=torch.float, device=self.device)
        # for i in range(num_envs):
            # object_indices = object_mask[i].nonzero().squeeze(1)
            # if len(object_indices):
                # indices = object_indices[torch.randint(len(object_indices), size=(num_object_points,))]
                # table_indices = table_mask[i].nonzero().squeeze(1)
                # table_indices_selected = table_indices[torch.randperm(len(table_indices))[:num_table_points[i]]]
                # indices = torch.cat([object_indices_selected, table_indices_selected], dim=0)
                # pcs[i, :] = images[i, indices, :3]
            # else:
                # invalid[i] = 1
        
        # compose pcs
        # pcs = pcs + torch.randn_like(pcs) * self.noise_std
        # pcs = torch.cat([
            # pcs, 
            # torch.zeros([pcs.shape[0], pcs.shape[1], 1], dtype=torch.float, device=self.device), 
        # ], dim=2)
        # return pcs #, invalid
 
 
class AcronymRenderer(Renderer):
    def __init__(self, image_width, image_height, intrinsics, gpu):
        self.device = torch.device(0)
        self.observer = Observer(image_width, image_height, intrinsics, gpu)
        
        #TODO: use acronym path
        with open('data/seen_similar_novel/acronym_used_ids.json','r') as f:
            obj_names = json.load(f)
        
        id = 1
        for object_code in tqdm(obj_names, desc='loading meshes'):
            object_mesh = tm.load(os.path.join('data/acronym/meshes/models', object_code, 'scaled.obj'), process=False).apply_scale(1.0)
            self.observer.add_object_mesh(object_code, object_mesh, id) # table is 0
            id += 1