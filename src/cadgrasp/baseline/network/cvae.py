import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance
from pytorch3d import transforms as pttf

from cadgrasp.baseline.utils.robot_model import RobotModel
from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.baseline.network.pointnet import PointNet


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, condition_size=1024):
        super().__init__()

        if conditional:
            assert condition_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, condition_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, condition_size)

    def forward(self, x, c=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #print('encoder', self.MLP)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1) # [B, 30+1024]
        #print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        #print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        #print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #print('decoder', self.MLP)

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            #print('z size {}'.format(z.size()))

        x = self.MLP(z)

        return x

class GraspCVAE(nn.Module):
    def __init__(self, obj_inchannel=3,
                 cvae_encoder_sizes=[1024, 512, 256], cvae_latent_size=64,
                 cvae_decoder_sizes=[1024, 256, 3+3+16], cvae_condition_size=512, obj_pc_num=10000):
        super(GraspCVAE, self).__init__()

        self.obj_inchannel = obj_inchannel
        self.cvae_encoder_sizes = cvae_encoder_sizes
        self.cvae_latent_size = cvae_latent_size
        self.cvae_decoder_sizes = cvae_decoder_sizes
        self.cvae_condition_size = cvae_condition_size
        self.cvae_encoder_sizes[0] = cvae_condition_size
        self.cvae_decoder_sizes[0] = cvae_condition_size
        self.obj_pc_num = obj_pc_num

        # self.obj_encoder = dict(
        #         mock=MockBackbone, 
        #         single=SinglePointNet, 
        #         seperate=SeperatePointNet, 
        #         sparseconv=SparseConv, 
        #     )[cfg.goal.vision_backbone_type](**cfg.goal.vision_backbone_parameters)
        self.hand_encoder = PointNet(pc_feature_dim=cvae_condition_size)
        self.cvae = VAE(encoder_layer_sizes=self.cvae_encoder_sizes,
                        latent_size=self.cvae_latent_size,
                        decoder_layer_sizes=self.cvae_decoder_sizes,
                        condition_size=self.cvae_condition_size)
        
        robot_name = 'leap_hand'
        urdf_path = f'robot_models/urdf/{robot_name}.urdf'
        meta_path = f'robot_models/meta/{robot_name}/meta.yaml'
        self.robot = RobotModel(
            urdf_path=urdf_path,
            meta_path=meta_path,
        )
        self.surface_points = self.robot.sample_surface_points(n_points=1024)

    def forward(self, obj_glb_feature, pose=None, inference=False):
        # B = hand_xyz.size(0)
        # device = hand_xyz.device
        # if self.training:
        #     return torch.zeros([B, 30], dtype=torch.float, device=device, requires_grad=True), torch.zeros([B, 30], dtype=torch.float, device=device), torch.zeros([B, 30], dtype=torch.float, device=device), torch.zeros([B, 64], dtype=torch.float, device=device), 
        # else:
        #     return torch.zeros([B, 30], dtype=torch.float, device=device)
        '''
        :param obj_pc: [B, 3+n, N1]
        :param hand_xyz: [B, 3, N2]
        :return: reconstructed hand params
        '''

        B = obj_glb_feature.size(0)
        if not inference:
            link_trans, link_rot = self.robot.forward_kinematics({k: pose['joints'][:, i] for i, k in enumerate(self.robot.joint_names)})
            if self.surface_points['hand_base_link'].device != obj_glb_feature.device:
                self.surface_points = {k: v.to(obj_glb_feature.device) for k, v in self.surface_points.items()}
            hand_xyz = torch.einsum('nab,nkb->nka', pose['rot'], torch.cat([torch.einsum('nab,kb->nka', link_rot[k], self.surface_points[k]) + link_trans[k][:, None] for k in link_trans.keys()], dim=1)) + pose['trans'][:, None]
            # obj_glb_feature, _ = self.obj_encoder(obj_pc) # [B, 512]
            hand_glb_feature, _ = self.hand_encoder(hand_xyz) # [B, 512]
            recon, means, log_var, z = self.cvae(hand_glb_feature, obj_glb_feature) # recon: [B, 30] or [B, 28]
            recon = recon.contiguous().view(B, -1)  # Flatten to [B, -1]
            loss, loss_dict = self.cal_loss(pose['trans_scale'], link_trans, link_rot, pose['trans'], pose['rot'], pose['joints'], hand_xyz, recon, pose['obj_pc'], means, log_var)
            # loss, loss_dict = self.cal_loss(pose['trans_scale'], link_trans, link_rot, pose['trans']*pose['trans_scale'], pose['rot'], pose['joints'], hand_xyz, recon, pose['obj_pc'], means, log_var)
            return loss, loss_dict
        else:
            # obj_glb_feature, _ = self.obj_encoder(obj_pc) # [B, 512]
            recon = self.cvae.inference(B, obj_glb_feature)
            recon = recon.contiguous().view(B, -1)  # ?
            rot = pttf.axis_angle_to_matrix(recon[:, 3:6])
            euc = torch.cat([recon[:, :3], recon[:, 6:]], dim=-1)
            return euc, rot
    
    def cal_loss(self, trans_scale, gt_link_trans, gt_link_rot, hand_trans, hand_rot, hand_qpos, hand_pc, hand_pose_pred, object_pc, mean, log_var, thres_contact=0.005, thres_dis=0.01, normalize_factor=200, 
            weight_trans=10., weight_rot=1., weight_qpos=1., weight_recon=1., weight_pen=300., weight_KLD=10., weight_cmap=1., weight_dis=0.):

        batch_size = len(hand_pose_pred)


        pred_rot = pttf.axis_angle_to_matrix(hand_pose_pred[:, 3:6])
        # self.hand_model.set_parameters_simple(hand_pose_pred[:, 6:])
        # self.hand_model.global_translation = hand_pose_pred[:, :3]
        # self.hand_model.global_rotation = pttf.axis_angle_to_matrix(hand_pose_pred[:, 3:6])
        link_trans, link_rot = self.robot.forward_kinematics({k: hand_pose_pred[:, 6+i] for i, k in enumerate(self.robot.joint_names)})
        hand_pc_pred = torch.einsum('nab,nkb->nka', pred_rot, torch.cat([torch.einsum('nab,kb->nka', link_rot[k], self.surface_points[k]) + link_trans[k][:, None] for k in link_trans.keys()], dim=1)) + hand_pose_pred[:, :3][:, None] # / trans_scale

        # hand_pc_pred = self.hand_model.get_surface_points()
        sel_idxs = torch.randint(0, object_pc.shape[1], (self.obj_pc_num,))
        distances = self.robot.cal_distance(gt_link_trans, gt_link_rot, hand_trans, hand_rot, object_pc[:, sel_idxs])
        # distances = self.robot.cal_distance(gt_link_trans, gt_link_rot, hand_trans/trans_scale, hand_rot, object_pc[:, sel_idxs])
        distances_pred = self.robot.cal_distance(link_trans, link_rot, hand_pose_pred[:, :3], pred_rot, object_pc[:, sel_idxs])
        # distances_pred = self.robot.cal_distance(link_trans, link_rot, hand_pose_pred[:, :3]/trans_scale, pttf.axis_angle_to_matrix(hand_pose_pred[:, 3:6]), object_pc[:, sel_idxs])

        # distances = hand['distances']  # signed squared distances from object_pc to hand, inside positive, outside negative
        # hand_pc = hand['surface_points']

        # distances_pred = hand_pred['distances']  # signed squared distances from object_pc to hand_pred, inside positve, outside negative
        # hand_pc_pred = hand_pred['surface_points']
        # contact_candidates_pred = hand_pred['contact_candidates']

        # loss params
        loss_trans = torch.nn.functional.mse_loss(hand_pose_pred[:, :3], hand_trans, reduction='none').sum(-1)
        hand_rot_aa = pttf.matrix_to_axis_angle(hand_rot)
        # hand_rot_aa_norm = hand_rot_aa.norm(dim=-1, keepdim=True)
        # hand_rot_aa = torch.where(hand_rot_aa_norm > torch.pi, -hand_rot_aa/hand_rot_aa_norm*(torch.pi*2-hand_rot_aa_norm), hand_rot_aa)
        loss_rot = torch.nn.functional.mse_loss(hand_pose_pred[:, 3:6], hand_rot_aa, reduction='none').sum(-1)
        loss_qpos = torch.nn.functional.mse_loss(hand_pose_pred[:, 6:], hand_qpos, reduction='none').sum(-1)

        # loss recon
        loss_recon, _ = chamfer_distance(hand_pc_pred, hand_pc, point_reduction='sum', batch_reduction=None)

        # loss KLD
        loss_KLD = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(-1)

        # loss pen
        loss_pen = (distances_pred - 0.002).clamp(min=0).sum(-1) * 0.001

        # loss cmap
        cmap = 2 - 2 * torch.sigmoid(normalize_factor * distances.abs())
        cmap_pred = 2 - 2 * torch.sigmoid(normalize_factor * distances_pred.abs())
        loss_cmap = torch.nn.functional.mse_loss(cmap, cmap_pred, reduction='none').sum(-1) * 0.001

        # loss dis
        # dis_pred = knn_points(object_pc, contact_candidates_pred).dists[:, :, 0]  # squared chamfer distance from object_pc to contact_candidates_pred
        # small_dis_pred = dis_pred < thres_dis ** 2
        # loss_dis = dis_pred[small_dis_pred].sum() / (small_dis_pred.sum() + 1e-4)

        # total loss
        # loss = weight_trans * loss_trans + weight_rot * loss_rot + weight_qpos * loss_qpos + weight_KLD * loss_KLD  #+ weight_dis * loss_dis
        # loss = weight_trans * loss_trans + weight_rot * loss_rot + weight_qpos * loss_qpos + weight_recon * loss_recon + weight_KLD * loss_KLD + weight_pen * loss_pen + weight_cmap * loss_cmap #+ weight_dis * loss_dis
        loss = weight_trans * loss_trans + weight_rot * loss_rot + weight_qpos * loss_qpos + weight_recon * loss_recon + weight_KLD * loss_KLD + weight_pen * loss_pen + weight_cmap * loss_cmap #+ weight_dis * loss_dis
        return loss, dict(loss_cvae=loss, loss_trans=loss_trans, loss_rot=loss_rot, loss_qpos=loss_qpos, loss_recon=loss_recon, loss_KLD=loss_KLD, loss_pen=loss_pen, loss_cmap=loss_cmap)