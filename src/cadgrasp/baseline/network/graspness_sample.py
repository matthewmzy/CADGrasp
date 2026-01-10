import numpy as np
import torch
from torch import nn
from pytorch3d.ops import sample_farthest_points, ball_query
from pytorch3d import transforms as pttf
from einops import rearrange, repeat
import warnings

from cadgrasp.baseline.network.backbones.backbones import get_backbone, get_feature
from cadgrasp.baseline.network.condition import ConditionalTransform
from cadgrasp.baseline.utils.util import proper_svd, to_voxel_center
from cadgrasp.baseline.network.diffusion import MLPWrapper, GaussianDiffusion1D
from cadgrasp.baseline.utils.dataset import Loader
from cadgrasp.baseline.network.cvae import GraspCVAE
from cadgrasp.baseline.utils.vis_plotly import Vis

class GraspnessSample(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.backbone = get_backbone(
            backbone_name=config.backbone, 
            feature_dim=config.feature_dim, 
            backbone_config=config.backbone_parameters[config.backbone])
        self.graspable = nn.Linear(config.feature_dim, 3)
        if config.type == 'graspness_diffusion':
            self.rot_type = config.diffusion.get('rot_type', 'svd')
            self.rot_dim = dict(
                svd=9,
                euler=3,
                quat=4,
                aa=3,
                sixd=6,
            )[self.rot_type]
            policy_mlp_parameters=dict(
                hidden_layers_dim=[512, 256], 
                output_dim=config.joint_num * config.dist_joint + 3 + self.rot_dim, 
                act='mish'
            )
            self.policy = MLPWrapper(channels=config.joint_num * config.dist_joint + 3 + self.rot_dim, feature_dim=config.feature_dim, **policy_mlp_parameters)
            self.diffusion = GaussianDiffusion1D(self.policy, config.diffusion)
        elif config.type == 'graspness_isa':
            assert config.dist_joint
            self.joint_mlp = ConditionalTransform(config.feature_dim, config.joint_num + 4 + 3)
        elif config.type == 'graspness_cvae':
            assert config.dist_joint
            self.grasp_cvae = GraspCVAE(**config.cvae)

        if not config.dist_joint:
            self.joint_mlp = ConditionalTransform(config.feature_dim + 9 + 3, config.joint_num)

        self.objectness_loss = nn.CrossEntropyLoss(reduction='none')
        self.graspness_loss = nn.SmoothL1Loss(reduction='none')
        self.joint_loss = nn.SmoothL1Loss(reduction='none')

        if not hasattr(config, 'voxel_size'):
            warnings.warn('voxel_size is not set, use 0.005 as default')
            self.config.voxel_size = 0.005
        self.register_buffer('mean', torch.zeros(9 + 3 + config.joint_num))
        self.register_buffer('std', torch.ones(9 + 3 + config.joint_num))

    def get_feature(self, data: dict):
        """
        extract feature from point clouds
        
        Args:
            point_clouds: (B, N, 3)
            coors, feats, quantize2original: cooresponding ME sparse tensor
        
        Returns:
            feature: (B, N, C)
        """
        feature = get_feature(
            backbone_name=self.config.backbone, 
            backbone=self.backbone, 
            data=data)
        return feature
    
    def pred_score(self, feature: torch.Tensor):
        """
        predict objectness and graspness score
        
        Args:
            feature: (B, N, C)
        
        Returns:
            objectness: (B, N, 2)
            graspness: (B, N)
        """
        graspable = self.graspable(feature)
        objectness = graspable[..., :2]
        graspness = graspable[..., 2]
        return objectness, graspness
    
    def sample_grasp(self, feature: torch.Tensor, seed_points: torch.Tensor, sample_num: int = 1, allow_fail: bool = False):
        """
        sample gripper pose from feature around seed points
        
        Args:
            feature: (B, C)
            seed_points: (B, 3)
        
        Returns:
            rot: (B, N, 3, 3)
            trans: (B, N, 3)
            width, depth: (B, N)
            log_prob: (B, N) or None
        """
        if self.config.type == 'graspness_diffusion':
            feature = repeat(feature, 'b c -> (b n) c', n=sample_num)
            samples, log_prob = self.diffusion.sample(cond=feature)
            rot, euc = samples[..., :self.rot_dim], samples[..., self.rot_dim:].reshape(len(seed_points), sample_num, -1)
            if self.rot_type == 'svd':
                rot = proper_svd(rot.reshape(-1, 3, 3)).reshape(-1, sample_num, 3, 3)
            elif self.rot_type == 'sixd':
                rot = pttf.rotation_6d_to_matrix(rot)
            elif self.rot_type == 'quat':
                rot = pttf.quaternion_to_matrix(rot)
            elif self.rot_type == 'aa':
                rot = pttf.axis_angle_to_matrix(rot)
            elif self.rot_type == 'euler':
                rot = pttf.euler_angles_to_matrix(rot, 'XYZ')
            rot = rot.reshape(-1, sample_num, 3, 3)
            log_prob = log_prob.reshape(-1, sample_num)
        elif self.config.type == 'graspness_isa':
            assert sample_num == 1
            est = self.joint_mlp(feature)
            quat, euc = est[:, :4], est[:, 4:]
            rot = pttf.quaternion_to_matrix(quat)
            rot, euc = rot[:, None], euc[:, None]
            log_prob = torch.zeros_like(euc[..., 0])
        elif self.config.type == 'graspness_cvae':
            assert sample_num == 1
            euc, rot = self.grasp_cvae(feature, inference=True)
            euc[:, :3] *= self.config.trans_scale
            euc[:, 3:] *= self.config.joint_scale
            rot, euc = rot[:, None], euc[:, None]
            log_prob = torch.zeros_like(euc[..., 0])

        if self.config.dist_joint:
            delta_trans = euc[..., :3]
            joints = euc[..., 3:] / self.config.joint_scale
        else:
            delta_trans = euc
            joints = self.joint_mlp(torch.cat([repeat(feature, 'b c -> (b n) c', n=sample_num), rot.reshape(-1, 9), delta_trans.reshape(-1, 3)], dim=-1))
            joints = rearrange(joints, '(b n) c -> b n c', n=sample_num) / self.config.joint_scale

        trans = self.to_voxel_center(seed_points[:, None]) + delta_trans / self.config.trans_scale
        return rot, trans, joints, log_prob
    
    def to_voxel_center(self, pc: torch.Tensor):
        """calculate the center of voxel corresponding to each point

        Args:
            pc (torch.Tensor): (..., 3)
        returns:
            voxel_center (torch.Tensor): (..., 3)
        """
        return to_voxel_center(pc, self.config.voxel_size)
    
    def get_euc_rot(self, data: dict):
        """
        get euclidean part and rotation matrix from data

        Args:
            point_clouds: (B, N, 3)
            trans: (B, K, 3)
            rot: (B, K, 3, 3)
            qpos: (B, K, J)
        
        Returns:
            rot: (B * K, 3, 3)
            euc: (B * K, 3)
        """
        batch_size = data['point_clouds'].shape[0]
        centers = data['trans']
        arange = repeat(torch.arange(batch_size, device=centers.device), 'n -> (n k)', k=centers.shape[1])
        indices = data['centers'].reshape(-1).long()
        points = data['point_clouds'][arange, indices]
        rot = rearrange(data['rot'], 'n k a b -> (n k) a b')
        trans = (rearrange(centers, 'n k d -> (n k) d') - self.to_voxel_center(points)) * self.config.trans_scale
        gt_joints = data['qpos']
        euc = trans
        if self.config.dist_joint:
            euc = torch.cat([euc, gt_joints.reshape(-1, gt_joints.shape[-1]) * self.config.joint_scale], dim=-1)
        return euc, rot
    
    def forward(self, data: dict):
        """
        calculate loss in training

        Args:
            point_clouds: (B, N, 3)
            coors, feats, quantize2original: cooresponding ME sparse tensor
            objectness: (B, N)
            graspness: (B, N)
            trans: (B, K, 3)
            rot: (B, K, 3, 3)
            qpos: (B, K, J)
            has_graspness: (B, 1)

        Returns:
            loss: scalar with gradient
            result_dict: (B,) 
                following this shape can makes it easier to log the distribution of each loss
        """
        # use sparse conv to extract features
        batch_size, point_num, _ = data['point_clouds'].shape
        feature = self.get_feature(data)

        # estimate grasp & obj score
        objectness, graspness = self.pred_score(feature)
        gt_objectness = data['objectness']
        gt_graspness = data['graspness']
        loss_objectness = self.objectness_loss(objectness.reshape(-1, 2), gt_objectness.reshape(-1)).reshape(-1, point_num).mean(dim=1)
        loss_graspness = self.graspness_loss(graspness * gt_objectness, gt_graspness * gt_objectness).sum(dim=1) / (gt_objectness.sum(dim=1) + 1e-6)
        loss_graspness = loss_graspness * data['has_graspness'].reshape(*loss_graspness.shape)
        acc_objectness = (objectness.argmax(dim=-1) == gt_objectness).float().mean(dim=-1)
        abs_graspness = torch.abs(graspness * gt_objectness - gt_graspness * gt_objectness).sum(dim=-1) / (gt_objectness.sum(dim=1) + 1e-6)

        # use flow to estimate gripper delta trans & rotation
        centers = data['trans']
        arange = repeat(torch.arange(batch_size, device=centers.device), 'n -> (n k)', k=centers.shape[1])
        indices = data['centers'].reshape(-1).long()
        sel_point_feature = feature[arange, indices]
        points = data['point_clouds'][arange, indices]
        trans = (rearrange(centers, 'n k d -> (n k) d') - self.to_voxel_center(points)) * self.config.trans_scale
        gt_joints = data['qpos']
        euc, rot = self.get_euc_rot(data)

        if self.config.type == 'graspness_diffusion':
            if self.rot_type == 'svd':
                rot_rep = rot.reshape(-1, 9)
            elif self.rot_type == 'sixd':
                rot_rep = pttf.matrix_to_rotation_6d(rot)
            elif self.rot_type == 'quat':
                rot_rep = pttf.matrix_to_quaternion(rot)
            elif self.rot_type == 'aa':
                rot_rep = pttf.matrix_to_axis_angle(rot)
            elif self.rot_type == 'euler':
                rot_rep = pttf.matrix_to_euler_angles(rot, 'XYZ')
            gt_goal = torch.cat([rot_rep, euc], dim=-1)
            loss_diffusion = self.diffusion(gt_goal, sel_point_feature)
        elif self.config.type == 'graspness_isa':
            est = self.joint_mlp(sel_point_feature)
            est_quat, est_euc = est[:, :4], est[:, 4:]
            est_rot = pttf.quaternion_to_matrix(est_quat)
            loss_euc = (est_euc - euc).abs().mean(dim=-1).reshape(batch_size, -1).mean(dim=-1)
            loss_quat = pttf.so3_relative_angle(est_rot, rot, eps=1e-2).reshape(batch_size, -1).mean(dim=-1)
        elif self.config.type == 'graspness_cvae':
            cvae_loss, cvae_losses = self.grasp_cvae(sel_point_feature, pose=dict(trans=trans/self.config.trans_scale, rot=rot, joints=gt_joints.reshape(-1, gt_joints.shape[-1]), trans_scale=self.config.trans_scale, obj_pc=repeat(data['point_clouds'], 'b n d -> (b k) n d', k=centers.shape[1])-self.to_voxel_center(points)[:, None]))

        if not self.config.dist_joint:
            est_joints = self.joint_mlp(torch.cat([sel_point_feature, rot.reshape(-1, 9), trans], dim=-1)).reshape(*gt_joints.shape)
            loss_joint = self.joint_loss(est_joints, gt_joints * self.config.joint_scale).mean(dim=-1).mean(dim=-1)
            abs_dis_joint = torch.abs(est_joints[..., 0] / self.config.joint_scale - gt_joints[..., 0]).mean(dim=-1)

        loss = self.config.weight.objectness * loss_objectness + \
               self.config.weight.graspness * loss_graspness
        
        result_dict = dict(loss_objectness=loss_objectness, 
                           loss_graspness=loss_graspness, 
                           acc_objectness=acc_objectness,
                           abs_graspness=abs_graspness)

        if not self.config.dist_joint:
            loss += self.config.weight.joint * loss_joint
            result_dict['loss_joint'] = loss_joint
            result_dict['abs_dis_joint'] = abs_dis_joint
        if self.config.type == 'graspness_diffusion':
            loss += self.config.weight.diffusion * loss_diffusion
            result_dict['loss_diffusion'] = loss_diffusion
        elif self.config.type == 'graspness_isa':
            loss += self.config.weight.euc * loss_euc + self.config.weight.quat * loss_quat
            result_dict['loss_euc'] = loss_euc
            result_dict['loss_quat'] = loss_quat
        elif self.config.type == 'graspness_cvae':
            cvae_loss = cvae_loss.reshape(batch_size, -1).mean(dim=-1)
            cvae_losses = {k: v.reshape(batch_size, -1).mean(dim=-1) for k, v in cvae_losses.items()}
            loss += cvae_loss
            result_dict.update(cvae_losses)
        result_dict['loss'] = loss

        return loss.mean(), result_dict 
    
    def sample_points(self, pc: torch.Tensor, graspable: torch.Tensor, k: int):
        if graspable.sum() == 0:
            raise ValueError('No graspable points')
        elif graspable.sum() <= k:
            indices = torch.randint(0, graspable.sum(), (k,), device=graspable.device)[None]
            seed_point = pc[graspable][indices[0]][None]
        else:
            seed_point, indices = sample_farthest_points(pc[graspable][None].contiguous(), K=k, random_start_point=True)
        return seed_point, indices
    
    def sample(self, data: dict, k: int, cate: bool = True, allow_fail: bool = False, graspness_scale=1, with_point=False, edge=None, with_graspness=False, ratio=0.005, near=False, with_score_parts=False):
        """
            Sample from the learned distribution

            data: dict with
                point_clouds: (B, N, 3)
                coors, feats, quantize2original: cooresponding ME sparse tensor
                seg: (B, N)
            k: sample number
            cate: whether to sample uniformly from each object category
            allow_fail: whether to allow the sampling to fail (e.g. not invertible matrix)
            edge: (B, N)

            returns
                rot: (B, K, 3, 3)
                trans: (B, K, 3)
                joints: (B, K, joint_num)
                score: (B, K)
                obj_indices: (B, K) which is -1 if not categorized
        """
        pc_cuda = data['point_clouds']
        b = pc_cuda.shape[0] 
        feature = self.get_feature(data)
        objectness, graspness = self.pred_score(feature)
        graspness = torch.where(objectness.argmax(dim=-1) == 1, graspness, torch.full_like(graspness, np.log(1e-3)))
        if edge is not None:
            graspness = torch.where(edge == 0, graspness, torch.full_like(graspness, np.log(1e-3)))

        features = []
        seed_points = []
        graspnesses = []
        obj_indices = []

        for i in range(b):
            obj_indices.append([])
            if cate:
                seg = data['seg'][i]
                obj_ids = [idx for idx in torch.unique(seg).tolist() if idx != 0]
                obj_num = len(obj_ids)
                obj_k = [k // obj_num for _ in range(obj_num)]
                for _ in range(k % obj_num):
                    obj_k[np.random.randint(obj_num)] += 1

                for j, obj_id in enumerate(obj_ids):
                    graspable = (seg == obj_id).to(objectness.device)
                    graspness_obj = graspness[i, graspable].sort(descending=True).values
                    threshold = graspness_obj[int(graspness_obj.size(0) * 0.05)]
                    graspable = (seg == obj_id).to(objectness.device) & (graspness[i] >= threshold)
                    seed_point, indices = self.sample_points(pc_cuda[i], graspable, obj_k[j])
                    features.append(feature[i, graspable][indices][0])
                    seed_points.append(seed_point[0])
                    graspnesses.append(graspness[i, graspable][indices][0])
                    obj_indices[-1] += [obj_id] * obj_k[j]
            else:
                if near:
                    K = 1000
                    graspable = graspness[i] >= graspness[i].sort(descending=True).values[int(graspness[i].size(0) * ratio)]
                    dists, idxs, nn = ball_query(pc_cuda[i, graspable][None], pc_cuda[i][None], radius=0.02, K=K)
                    graspness_around = torch.where(idxs.reshape(-1) != -1, graspness[i][idxs.reshape(-1)], torch.full_like(idxs.reshape(-1), np.log(1e-3)).float()).reshape(-1, K)
                    good = graspness_around > graspness_around.sort(descending=True).values[torch.arange(len(idxs[0]), device=idxs.device), ((idxs[0]!=-1).sum(-1)*0.1).long()][:, None]
                    new_idxs = idxs.reshape(-1)[good.reshape(-1)]
                    # level = (graspness_around > graspness[i][graspable][:, None]).float().sum(-1) / (idxs[0] != -1).sum(-1)
                    # graspable[graspable.nonzero().reshape(-1)[level > 0.33]] = 0
                    # graspable[graspness[i] >= graspness[i].sort(descending=True).values[int(graspness[i].size(0) * 0.0025)]] = 1
                    if len(new_idxs) != 0:
                        graspable[:] = 0
                        graspable[new_idxs] = 1
                    # graspable[idxs[0][torch.arange(idxs[0].size(0), device=idxs.device), graspness_around.argmax(dim=-1)]] = 1
                else:
                    graspable = graspness[i] >= graspness[i].sort(descending=True).values[int(graspness[i].size(0) * ratio)]
                    # graspable = torch.logical_and(graspness[i] > np.log(1e-3), graspable)
                    graspable = graspness[i] > np.log(1e-2)
                    if graspable.sum() == 0:
                        graspable = graspness[i] >= graspness[i].sort(descending=True).values[int(graspness[i].size(0) * ratio)]
                seed_point, indices = self.sample_points(pc_cuda[i], graspable, k)
                features.append(feature[i, graspable][indices][0])
                seed_points.append(seed_point[0])
                graspnesses.append(graspness[i, graspable][indices][0])
                obj_indices[-1] += [-1] * k

        features, seed_points = torch.cat(features), torch.cat(seed_points)
        rot, trans, joints, log_prob = self.sample_grasp(features, seed_points, sample_num=1, allow_fail=allow_fail)
        rot, trans, joints, log_prob = rot.reshape(b, k, 3, 3), trans.reshape(b, k, 3), joints.reshape(b, k, -1), log_prob.reshape(b, k)
        graspnesses = torch.cat(graspnesses, dim=0).reshape(b, k)

        normalize_log_prob = (log_prob - log_prob.mean()) / log_prob.std()
        normalize_graspness = (graspnesses - graspnesses.mean()) / graspnesses.std()

        score = log_prob + graspnesses * graspness_scale
        score = score.nan_to_num(nan=-1e6)

        # idxs = score[0].argsort(descending=True)[::len(score[0])//10]
        # print(score[0][idxs])
        # print(graspnesses[0][idxs])
        # print(log_prob[0][idxs])

        # log_prob_thresh = log_prob.reshape(-1).sort().values[-int(log_prob.shape[1] * 0.1)]
        # graspness_thresh = graspnesses.reshape(-1).sort().values[-int(log_prob.shape[1] * 0.1)]
        # mask = (log_prob > log_prob_thresh) & (graspnesses > graspness_thresh)
        # score = torch.where(mask, score, torch.full_like(score, -1e6))

        obj_indices = torch.tensor(obj_indices).to(rot.device)

        result = [rot, trans, joints, score, obj_indices]
        if with_score_parts:
            result.append(graspnesses)
            result.append(log_prob)
        if with_point:
            result.append(seed_points)
        if with_graspness:
            result.append(graspable.float())
        return result
