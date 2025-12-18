import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

from PIL import Image
import scipy.io as scio
import random
import collections.abc as container_abcs
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
from pytorch3d import transforms as pttf

from src.utils.pc import depth_image_to_point_cloud, get_workspace_mask
from src.utils.util import to_voxel_center
from src.utils.pose_refine import PoseRefine
from src.utils.vis_plotly import Vis
from src.utils.robot_model import RobotModel

class Loader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iter = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        return data
    
class GraspNetDataset(Dataset):
    def __init__(self, 
                 config: dict, 
                 split: str,
                 is_train: bool = False,
                 is_eval: bool = False,
    ):
        self.full_config = config
        self.config = config = config.data
        self.is_train = is_train
        self.is_eval = is_eval

        splits = dict(
            single=range(1),
            half=range(45),
            train=range(100),
            train_1=range(90),
            val=range(90, 100),
            test=range(100, 190),
            test_seen=range(100, 130),
            test_similar=range(130, 160),
            test_novel=range(160, 190),
        )
        self.scene_id = [f'scene_{str(x).zfill(4)}' for x in splits[split.split('-')[0]][::self.config.scene_fraction if is_train else 1]]
        ann_id = range(1 if split == 'single' else 256)
        self.views = []
        for scene in self.scene_id:
            for i in ann_id:
                self.views.append((scene, i))
        if config.robot == 'gripper': 
            self.refiner = PoseRefine()
        else:
            self.robot_model = RobotModel(os.path.join('robot_models', 'urdf', config.robot + '.urdf'), os.path.join('robot_models', 'meta', config.robot, 'meta.yaml'))
            self.joint_names = self.robot_model.joint_names
        self.cates = ['orig']
        if is_train:
            if split.split('-')[1] == 'part':
                self.cates += ['part']
    
    def __len__(self):
        return 100000 if self.is_train else len(self.views)
    
    def __getitem__(self, dataset_idx: int):
        cate = random.choice(self.cates)
        if cate == 'orig':
            if not self.is_eval:
                scene, view = random.choice(self.views)
            else:
                scene, view = self.views[dataset_idx]
        elif cate == 'part':
            orig_scenes = list(range(100))[::self.config.scene_fraction]
            scene = f'scene_{1000 + random.choice(orig_scenes) * 75 + random.randint(0, 74)}'
            view = random.randint(0, 255)
        try:
            str_view = str(view).zfill(4)
            suffix = '_gt' if self.config.render else ''

            # loading
            path = os.path.join('data', 'scenes', scene, self.config.camera)
            depth = np.array(Image.open(os.path.join(path, 'depth'+suffix, str_view + '.png')))
            seg = np.array(Image.open(os.path.join(path, 'label'+suffix, str_view + '.png')))
            try:
                edge = np.array(Image.open(os.path.join(path, 'edge'+suffix, str_view + '.png')))
            except:
                edge = None
            meta = scio.loadmat(os.path.join(path, 'meta', str_view + '.mat'))
            instrincs = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
            camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))

            # get pointcloud in workspace in camera frame
            cloud = depth_image_to_point_cloud(depth, instrincs, factor_depth)
            depth_mask = (depth > 0)
            trans = np.dot(align_mat, camera_poses[view])
            if not seg.any():
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            workspace_mask = get_workspace_mask(cloud, seg, trans)
            mask = (depth_mask & workspace_mask)
            cloud = cloud[mask]
            seg = seg[mask]

            # random sample
            idxs = np.random.choice(len(cloud), self.config.num_points, replace=True)
            cloud = cloud[idxs]
            seg = seg[idxs]

            if self.is_eval:
                ret_dict = {
                    'scene': np.array([int(scene.split('_')[-1])]),
                    'view': np.array([view]),
                    'point_clouds': cloud.astype(np.float32), # (N, 3)
                    'coors': cloud.astype(np.float32) / self.config.voxel_size, # (N, 3)
                    'feats': np.ones_like(cloud).astype(np.float32), # (N, 3)
                    'seg': seg.astype(np.int64), # (N,)
                }
                if edge is not None:
                    ret_dict['edge'] = edge[mask][idxs]
                return ret_dict

            frac_suffix = '' if self.config.fraction == 1 else f'_{self.config.fraction}'
            graspness_path = os.path.join('data', self.config.graspness_data+frac_suffix, scene, self.config.camera, str_view + '.npy')
            if os.path.exists(graspness_path):
                graspness = np.load(graspness_path)
                graspness = graspness.reshape(-1)
                graspness = graspness[idxs]
                graspness = np.log(graspness + 1e-3)
                has_graspness = 1
            else:
                graspness = np.zeros((len(cloud),), dtype=np.float32)
                has_graspness = 0

            # load poses from world frame and transform to camera frame
            if self.config.robot == 'gripper':
                poses_6d = np.load(os.path.join('data', 'gripper_grasps', scene, self.config.camera, 'poses.npy'))[::self.config.fraction]
                grasp_points = np.load(os.path.join('data', 'gripper_grasps', scene, self.config.camera, 'points.npy'))[::self.config.fraction]

                assert self.config.sample_total >= self.config.k
                if self.config.resample:
                    can_grasp_ids = np.unique(poses_6d[:, -1])
                    rand_idxs = np.random.randint(0, len(can_grasp_ids), self.config.sample_total)
                    samples = []
                    point_samples = []
                    for i, idx in enumerate(can_grasp_ids):
                        num = (rand_idxs == i).sum()
                        obj_poses_6d = poses_6d[poses_6d[:, -1] == idx]
                        obj_rand_idxs = np.random.choice(len(obj_poses_6d), num, replace=True)
                        samples.append(obj_poses_6d[obj_rand_idxs])
                        point_samples.append(grasp_points[poses_6d[:, -1] == idx][obj_rand_idxs])
                    poses_6d = np.concatenate(samples)
                    point_samples = np.concatenate(point_samples)
                    permute = np.random.permutation(len(poses_6d))
                    poses_6d = poses_6d[permute]
                    point_samples = point_samples[permute]
                else:
                    idxs = np.random.choice(len(poses_6d), self.config.sample_total, replace=True)
                    poses_6d = poses_6d[idxs]
                    point_samples = grasp_points[idxs]
                rot = poses_6d[:, -13:-4].reshape(-1, 3, 3)
                trans = poses_6d[:, -4:-1]

            else:
                data_root = 'data'
                assert self.config.resample
                grasp_files = os.listdir(os.path.join(data_root, 'dex_grasps_new'+frac_suffix, scene, self.config.robot))
                if len(grasp_files) == 0:
                    return self.__getitem__(random.randint(0, self.__len__() - 1))
                rand_idxs = np.random.randint(0, len(grasp_files), self.config.sample_total)
                samples = []
                for i, f in enumerate(grasp_files):
                    num = (rand_idxs == i).sum()
                    grasps = np.load(os.path.join(data_root, 'dex_grasps_new'+frac_suffix, scene, self.config.robot, f))
                    sel_idxs = np.random.choice(len(grasps['point']), num, replace=True)
                    samples.append({k: grasps[k][sel_idxs] for k in grasps.keys()})
                permute = np.random.permutation(self.config.sample_total)
                samples = {k: np.concatenate([sample[k] for sample in samples])[permute] for k in samples[0].keys()}
                rot = samples['rotation']
                trans = samples['translation']

            new_rot = np.einsum('ji,njk->nik', camera_poses[view, :3, :3], rot)
            new_trans = np.einsum('ji,nj->ni', camera_poses[view, :3, :3], trans - camera_poses[view, :3, 3])

            grasp_points = point_samples if self.config.robot == 'gripper' else samples['point']
            grasp_points = np.einsum('ba,nb->na', camera_poses[view, :3, :3], grasp_points - camera_poses[view, :3, 3])
            centers = np.zeros((self.config.sample_total,),)
            available = []
            dis = []
            for i in range(len(centers)):
                if len(available) >= self.config.k:
                    break
                try:
                    nearest_idx = np.linalg.norm(cloud - grasp_points[i], axis=1).argmin()
                    if np.linalg.norm(cloud[nearest_idx] - grasp_points[i]) > self.config.max_point_dis:
                        raise Exception
                    centers[i] = nearest_idx
                    dis.append(np.linalg.norm(cloud[nearest_idx] - grasp_points[i]))
                    available.append(i)
                except:
                    pass
            if len(available) == 0:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            dis = np.array(dis)
            indices = np.random.choice(np.array(available), self.config.k, replace=True)
            if self.config.robot == 'gripper':
                poses_6d = poses_6d[indices]
            else:
                qpos = np.stack([samples[j] for j in self.joint_names], axis=-1)
                qpos = qpos[indices]

            new_rot = new_rot[indices]
            new_trans = new_trans[indices]
            centers = centers[indices]

            ret_dict = {
                'point_clouds': cloud.astype(np.float32), # (N, 3)
                'coors': cloud.astype(np.float32) / self.config.voxel_size, # (N, 3)
                'feats': np.ones_like(cloud).astype(np.float32), # (N, 3)
                'seg': seg.astype(np.int64), # (N,)
                'objectness': (seg > 0).astype(np.int64), # (N,)
                'graspness': graspness.astype(np.float32), # (N,)
                'rot': new_rot.astype(np.float32), # (K, 3, 3)
                'trans': new_trans.astype(np.float32), # (K, 3)
                'centers': centers.astype(np.float32), # (K,)
                'has_graspness': np.array([has_graspness]), # (1,)
            }

            if self.config.robot == 'gripper':
                ret_dict.update({
                    'qpos': poses_6d[:, [1]].astype(np.float32), # (K, 1)
                })
            else:
                ret_dict.update({
                    'qpos': qpos.astype(np.float32), # (K, J)
                })
            
            if self.is_train:
                ret_dict = self.augment_data(ret_dict)

            return ret_dict
        except Exception as e:
            print('Unknow error in loading dataset')
            print(f'{cate} {scene} {view}')
            print(e)
            return self.__getitem__(random.randint(0, self.__len__() - 1))
    
    def augment_data(self, ret_dict: dict):
        """
            Random rotate the point cloud and the gripper pose
        """

        cloud = ret_dict['point_clouds']
        rot = ret_dict['rot']
        trans = ret_dict['trans']

        theta = np.random.rand() * 2 * np.pi
        rotmat = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]).astype(np.float32)
        cloud = np.einsum('ij,nj->ni', rotmat, cloud)
        trans = np.einsum('ij,nj->ni', rotmat, trans)
        rot = np.einsum('ij,njk->nik', rotmat, rot)
        
        ret_dict['point_clouds'] = cloud
        ret_dict['rot'] = rot
        ret_dict['trans'] = trans
        ret_dict['coors'] = ret_dict['point_clouds'] / self.config.voxel_size
        return ret_dict
    
def get_sparse_tensor(pc: torch.tensor, voxel_size: float):
    """
        pc: (B, N, 3)
        return dict(point_clouds, coors, feats, quantize2original)
    """
    coors = pc / voxel_size
    feats = np.ones_like(pc)
    coordinates_batch, features_batch = ME.utils.sparse_collate([coor for coor in coors], [feat for feat in feats])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch, return_index=True, return_inverse=True)
    return dict(point_clouds=pc, coors=coordinates_batch, feats=features_batch, quantize2original=quantize2original)

# some magic to get MinkowskiEngine sparse tensor
def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, original2quantize, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "original2quantize": original2quantize, 
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res

