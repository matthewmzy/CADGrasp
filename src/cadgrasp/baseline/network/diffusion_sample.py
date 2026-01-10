import torch
from torch import nn
from pytorch3d.ops import knn_points
from pytorch3d import transforms as pttf
from einops import rearrange, repeat

from cadgrasp.baseline.network.condition import ConditionalTransform
from cadgrasp.baseline.utils.util import proper_svd
from cadgrasp.baseline.network.diffusion import MLPWrapper, GaussianDiffusion1D, SinusoidalPosEmb
from cadgrasp.baseline.network.backbones.backbones import get_backbone, get_feature

class DiffusionSample(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.type = config.type

        policy_mlp_parameters=dict(
            hidden_layers_dim=[512, 256], 
            output_dim=config.joint_num * config.dist_joint + 3 + 9, 
            act='mish'
        )
        self.policy = MLPWrapper(channels=config.joint_num * config.dist_joint + 3 + 9, feature_dim=config.feature_dim, **policy_mlp_parameters)
        self.diffusion = GaussianDiffusion1D(self.policy, config.diffusion)
        self.backbone = get_backbone(
            backbone_name=config.backbone, 
            feature_dim=config.feature_dim, 
            backbone_config=config.backbone_parameters[config.backbone])
        
        self.joint_loss = nn.SmoothL1Loss(reduction='none')
        if not config.dist_joint:
            self.joint_mlp = ConditionalTransform(config.feature_dim + 9 + 3, config.joint_num)

    def set_norm(self, x):
        return
    
    def forward(self, data: dict):
        b, k, _ = data['trans'].shape
        feature = self.get_feature(data)
        feat = repeat(feature, 'b c -> (b k) c', k=k)
        trans = rearrange(data['trans'], 'b k c -> (b k) c')
        rot = rearrange(data['rot'], 'b k x y -> (b k) x y')
        joints = rearrange(data['qpos'], 'b k c -> (b k) c')

        gt_goal = torch.cat([rot.reshape(-1, 9), trans], dim=-1)
        if self.config.dist_joint:
            gt_goal = torch.cat([gt_goal, joints], dim=-1)
        loss_diffusion = self.diffusion(gt_goal, feat)

        if not self.config.dist_joint:
            est_joints = self.joint_mlp(torch.cat([feat, rot.reshape(-1, 9), trans], dim=-1)).reshape(*joints.shape)
            loss_joint = self.joint_loss(est_joints, joints * self.config.joint_scale).mean(dim=-1).mean(dim=-1)
            abs_dis_joint = torch.abs(est_joints[..., 0] / self.config.joint_scale - joints[..., 0]).mean(dim=-1)
        else:
            loss_joint = 0*loss_diffusion

        loss = self.config.weight.joint * loss_joint + self.config.weight.diffusion * loss_diffusion

        return_dict = dict(loss=loss,
                           loss_diffusion=loss_diffusion,
                           loss_joint=loss_joint)
        
        return loss.mean(), return_dict

    def sample(self, data: dict, sample_num: int = 1, with_point: bool = False, **useless):
        b = data['point_clouds'].shape[0]
        feature = self.get_feature(data)
        feature = repeat(feature, 'b c -> (b k) c', k=sample_num)
        samples, log_prob = self.diffusion.sample(cond=feature)
        rot, euc = samples[..., :9].reshape(-1, 3, 3), samples[:, 9:].reshape(len(feature), -1)

        if self.config.dist_joint:
            trans = euc[:, :3]
            joint = euc[:, 3:]
        else:
            trans = euc
            joint = self.joint_mlp(torch.cat([feature, rot.reshape(-1, 9), trans], dim=-1)).reshape(b, sample_num, -1)

        if with_point:
            return rot.reshape(b, -1, 3, 3), trans.reshape(b, -1, 3), joint.reshape(b, sample_num, -1), log_prob.reshape(b, sample_num), torch.full((b, sample_num), -1., device=feature.device), torch.zeros((b * sample_num, 3), device=feature.device)
        else:
            return rot.reshape(b, -1, 3, 3), trans.reshape(b, -1, 3), joint.reshape(b, sample_num, -1), log_prob.reshape(b, sample_num), torch.full((b, sample_num), -1., device=feature.device)
    
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