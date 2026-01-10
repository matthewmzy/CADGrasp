import torch
import json
import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pdb
from plotly import graph_objects as go
from termcolor import cprint
from tqdm import tqdm

from cadgrasp.optimizer.HandModel import HandModel, get_handmodel
from cadgrasp.optimizer.visualize import plot_point_cloud, plot_point_cloud_dis
from cadgrasp.optimizer.utils_hand_meta import *
from cadgrasp.optimizer.rot6d import add_transform_perturbation
from cadgrasp.optimizer.ibs_func import IBSFilter, visualize_point_clouds
from LASDiffusion.utils.visualize_ibs_vox import get_ibs_pc
from LASDiffusion.utils.visualize_ibs_vox import get_ibs_pc, devoxelize

import torch.nn.functional as F


class IBSAdam:
    def __init__(self, hand_name, parallel_num=10, running_name=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 verbose_energy=False, ik_step=0):
        self.running_name = running_name
        self.device = device
        self.hand_name = hand_name
        self.parallel_num = parallel_num

        self.verbose_energy = verbose_energy

        self.global_step = None
        self.q_current = None
        self.energy = None

        self.ibs_size = None
        self.ibs_pcd = None         # BxNx3
        self.ibs_hand_dis = None    # N

        self.q_global = None
        self.q_local = None
        self.optimizer = None
        self.scheduler = None

        self.ik_step = ik_step

    def get_initial_pose(self):
        transform = torch.inverse(self.handmodel.w2h_trans_meta).to(self.device)
        moveback_trans = torch.eye(4, device=self.device)
        moveback_trans[2][3] = 0.2
        moveback_trans.unsqueeze(0).repeat((transform.shape[0],1,1))
        transform = torch.matmul(transform, moveback_trans)
        transform = add_transform_perturbation(transform, max_translation=0.03, max_rotation=10)
        translation, rotation = transform[:,:3,3], transform[:,:3,:3]
        rotation = torch.transpose(rotation, 1, 2).reshape((self.num_particles,-1))[:,:6]
        joint_angles = torch.tensor([[0, 0.3, 0.3, 0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3, 0., 1.3, 0, 0, 0]], device=self.device).repeat((self.num_particles,1))
        initial_pose = torch.cat([translation, rotation, joint_angles], dim=1)

        return initial_pose

    def reset(self, ibs_triplets, running_name, cone_viz_num=0, cone_mu=1.0, filt_or_not=True, verbose=False):
        self.running_name = running_name
        self.ibs_size = 1024
        self.cont_size = 80
        self.thumb_cont_size = 20
        self.global_step = 0
        
        self.ibs_pcd = []
        self.cont_pcd = []
        self.thumb_cont_pcd = []
        self.distals_from_ibs = []
    
        for ibs_occu, ibs_cont, ibs_thumb_cont in ibs_triplets:
            
            # 下采样到 self.ibs_size
            ibs_occu = ibs_occu[np.random.choice(ibs_occu.shape[0], self.ibs_size, replace=(ibs_occu.shape[0] < self.ibs_size))]
            if ibs_cont.shape[0] == 0:
                ibs_cont = ibs_occu[:self.cont_size]
                cprint(f"[WARNING] No contact points found, using meaningless points as contact points", 'red')
            else:
                ibs_cont = ibs_cont[np.random.choice(ibs_cont.shape[0], self.cont_size, replace=(ibs_cont.shape[0] < self.cont_size))]
            if ibs_thumb_cont.shape[0] == 0:
                ibs_thumb_cont = ibs_cont[:self.thumb_cont_size]
                cprint(f"[WARNING] No thumb contact points found, using meaningless points as thumb contact points", 'red')
            else:
                ibs_thumb_cont = ibs_thumb_cont[np.random.choice(ibs_thumb_cont.shape[0], self.thumb_cont_size, replace=(ibs_thumb_cont.shape[0] < self.thumb_cont_size))]
            self.ibs_pcd += [torch.from_numpy(ibs_occu).float().to(self.device)]*self.parallel_num
            self.cont_pcd += [torch.from_numpy(ibs_cont).float().to(self.device)]*self.parallel_num
            self.thumb_cont_pcd += [torch.from_numpy(ibs_thumb_cont).float().to(self.device)]*self.parallel_num

        self.num_particles = len(ibs_triplets)*self.parallel_num

        self.handmodel = get_handmodel(self.hand_name, self.num_particles, self.device, hand_scale=1.)
        self.q_joint_lower = self.handmodel.revolute_tendon_q_lower.detach()
        self.q_joint_upper = self.handmodel.revolute_tendon_q_upper.detach()

        self.ibs_pcd = torch.stack(self.ibs_pcd, dim=0)  # (num_particles, 1024, 6)
        self.cont_pcd = torch.stack(self.cont_pcd, dim=0)  # (num_particles, 64, 6)
        self.thumb_cont_pcd = torch.stack(self.thumb_cont_pcd, dim=0)  # (num_particles, 20, 6)
        self.distals_from_ibs = torch.from_numpy(np.array(self.distals_from_ibs)).float().to(self.device)  # (num_particles, 3, 3)

        self.initial_pose = self.get_initial_pose()
        self.q_current = self.initial_pose.clone()

        """
        visualize the hand and ibs
        """
        # hand_plotly_data = self.handmodel.get_plotly_data(self.q_current,0)
        # ibs_plotly_data = plot_point_cloud(self.ibs_pcd[0].cpu().numpy(), color='red')
        # cont_plotly_data = plot_point_cloud(self.cont_pcd[0].cpu().numpy(), color='green')
        # thumb_plotly_data = plot_point_cloud(self.thumb_cont_pcd[0].cpu().numpy(), color='blue')
        # fig = go.Figure(data=hand_plotly_data+[ibs_plotly_data, cont_plotly_data, thumb_plotly_data])
        # fig.show()

    def set_opt_weight(self, w_trans, w_rot, w_joint, learning_rate, decay_every, lr_decay):
        # Set weights
        self.translation_weight = w_trans
        self.rotation_weight = w_rot
        self.joint_weight = w_joint

        # Create leaf tensors by detaching and cloning
        self.test_translation = self.q_current[:, :3].detach().clone() * self.translation_weight
        self.test_rotation = self.q_current[:, 3:9].detach().clone() * self.rotation_weight
        self.test_joint = self.q_current[:, 9:].detach().clone() * self.joint_weight

        # Set requires_grad for leaf tensors
        self.test_translation.requires_grad = True
        self.test_rotation.requires_grad = True
        self.test_joint.requires_grad = True

        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW([self.test_translation, self.test_rotation, self.test_joint], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_every, gamma=lr_decay)

    def contact_match_energy(self):
        '''
        params: 
            q_current:          [num_particles, 3 + 6 + num_joints]
            palmar_points:      [num_particles, num_palmar_points, 3]
            ibs_pcd(……):        [num_particles, ibs_size, 6]
        '''
        palmar_points, _, palmar_finger_points, _, palmar_thumb_points, _ = self.handmodel.get_surface_points_and_normals(palmar=True)
        energy = torch.zeros((self.num_particles, self.ibs_size), device=self.device)


        # 每个contact点找最近的掌内侧手指点
        dis_mat = torch.cdist(self.cont_pcd[:,:,:3], palmar_finger_points)  # [num_particles, contact_num, palmar_finger_points_num]
        min_dis_1, _ = torch.min(dis_mat, dim=2)  # [num_particles, contact_num]
        # 每个掌内侧手指点找最近的contact点
        min_dis_2, _ = torch.min(dis_mat, dim=1)  # [num_particles, palmar_finger_points_num]

        dis_mat1 = torch.cdist(self.thumb_cont_pcd[:,:,:3], palmar_thumb_points)  # [num_particles, contact_num, palmar_thumb_points_num]
        min_dis_3, _ = torch.min(dis_mat1, dim=2)  # [num_particles, contact_num]
        min_dis_4, _ = torch.min(dis_mat1, dim=1)  # [num_particles, palmar_thumb_points_num]

        # energy_1 = torch.square(min_dis_1)*50
        # energy_3 = torch.square(min_dis_3)*50
        # energy_1 = (torch.exp(min_dis_1*30)-1)/10
        # energy_3 = (torch.exp(min_dis_3*30)-1)/10
        energy_1 = min_dis_1*10
        energy_3 = min_dis_3*10
        energy_2 = min_dis_2
        energy_4 = min_dis_4

        energy_1 = energy_1.mean(dim=1)
        energy_2 = energy_2.mean(dim=1)
        energy_3 = energy_3.mean(dim=1)
        energy_4 = energy_4.mean(dim=1)

        return energy_1, energy_2, energy_3, energy_4
    
    def finger_distals_match_energy(self):
        '''
        params: 
            distal_points           [num_particles, 3, 3]
            ibs_data                [num_particles, ibs_size, 3]
        '''
        distal_points = self.handmodel.get_finger_distals()
        diffs = torch.norm(distal_points - self.distals_from_ibs, dim=2)
        energy = diffs.mean(dim=1)
        # energy = diffs[:,0]
        return energy

    def joint_limits_energy(self):
        '''
        params:
            q_current:          [num_particles, 3 + 6 + num_joints]
            q_joint_lower:      [num_joints]
            q_joint_upper:      [num_joints]
        '''
        joint_angles = self.q_current[:,9:]

        energy = F.relu(joint_angles - self.q_joint_upper).sum(dim=1) + F.relu(self.q_joint_lower - joint_angles).sum(dim=1)

        return energy

    def self_penetration_energy(self):
        energy = self.handmodel.get_self_penetration()
        return energy

    def penetration_energy_sdf(self):
        ibs_pcd = self.ibs_pcd[:,:,:3]  # [num_particles, ibs_size, 3]
        dis = -self.handmodel.calculate_distances(ibs_pcd)
        dis = torch.clamp(dis, min=0)
        # dis = torch.exp(1000 * dis) - torch.ones_like(dis)
        dis = torch.exp(1000 * (dis-0.01)) - torch.exp(torch.ones_like(dis)*(-10))
        energy = dis.mean(dim=1)
        return energy
    
    def penetration_energy(self, thre=0.02):
        surface_points, _ = self.handmodel.get_surface_points_and_normals(palmar=False)     
        ibs_pcd = self.ibs_pcd[:,:,:3]  # [num_particles, ibs_size, 3]
        ibs_hand_uv = self.ibs_pcd[:,:,3:]  # [num_particles, ibs_size, 3]
        dis_mat = torch.cdist(ibs_pcd, surface_points)  # [num_particles, ibs_size, surface_points_num]
        vec_map = surface_points.unsqueeze(1) - ibs_pcd.unsqueeze(2)  # [num_particles, ibs_size, surface_points_num, 3]
        ibs_hand_uv = ibs_hand_uv.unsqueeze(2)  # [num_particles, ibs_size, 1, 3]
        cos_dis = (vec_map * ibs_hand_uv).sum(dim=3)  # [num_particles, ibs_size, surface_points_num]
        cos_dis = torch.clamp(cos_dis, max=0.005)
        cos_dis[dis_mat>thre] = 0
        cos_dis = -cos_dis
        energy = cos_dis.sum(dim=2).mean(dim=1)
        return energy
    
    def penetration_energy_improved(self):
        """
        计算穿模惩罚：惩罚手表面点出现在 IBS 面法向量背侧的情况。
        """
        # 获取手表面点（非掌侧）和 IBS 点云
        surface_points, _ = self.handmodel.get_surface_points_and_normals(palmar=False)  # [num_particles, surface_points_num, 3]
        ibs_pcd = self.ibs_pcd[:, :, :3]  # [num_particles, ibs_size, 3]
        ibs_hand_uv = self.ibs_pcd[:, :, 3:]  # [num_particles, ibs_size, 3]

        # 计算手表面点到 IBS 点的最小距离和对应 IBS 点索引
        dis_mat = torch.cdist(surface_points, ibs_pcd)  # [num_particles, surface_points_num, ibs_size]
        min_dis, min_idx = torch.min(dis_mat, dim=2)  # [num_particles, surface_points_num]

        # 收集每个表面点对应的最近 IBS 点的法向量
        batch_idx = torch.arange(self.num_particles, device=self.device).unsqueeze(1)  # [num_particles, 1]
        nearest_ibs_uv = ibs_hand_uv[batch_idx, min_idx]  # [num_particles, surface_points_num, 3]

        # 计算表面点到最近 IBS 点的向量
        nearest_ibs_pcd = ibs_pcd[batch_idx, min_idx]  # [num_particles, surface_points_num, 3]
        vec_to_surface = surface_points - nearest_ibs_pcd  # [num_particles, surface_points_num, 3]

        # 计算点积，判断是否在背侧
        dot_products = (vec_to_surface * nearest_ibs_uv).sum(dim=2)  # [num_particles, surface_points_num]
        penetration_mask = dot_products < 0  # 在背侧（点积 < 0）

        # 计算惩罚：仅对背侧点施加惩罚，惩罚值与点积和距离相关
        penalties = torch.zeros_like(dot_products)  # [num_particles, surface_points_num]
        penalties[penetration_mask] = -dot_products[penetration_mask]  # 负点积转为正惩罚
        # penalties = penalties * torch.exp(-min_dis)  # 距离越近，惩罚越大

        # 汇总能量：对每个粒子平均惩罚
        energy = penalties.sum(dim=1) / surface_points.shape[1]  # [num_particles]

        return energy

    def punish_transform(self):
        # 惩罚self.q_current前9维相对self.initial_pose的变化
        transform_current = self.q_current[:,:9]
        transform_initial = self.initial_pose[:,:9]
        energy = torch.norm(transform_current - transform_initial, dim=1)
        return energy

    def compute_energy(self, energy_dict=None):
        E_joint = self.joint_limits_energy()*5
        E_spen = self.self_penetration_energy()
        E_cont_1, E_cont_2, E_cont_3, E_cont_4 = self.contact_match_energy()
        E_cont_1 *= 8   # ibs点拉最近的手指点
        E_cont_2 *= 2   # 手指点拉最近的ibs
        E_cont_3 *= 10   # ibs点拉最近的拇指点
        E_cont_4 *= 0   # 拇指点拉最近的ibs
        # E_pen = self.penetration_energy_sdf()*100000 
        # E_pen = self.penetration_energy()*5000
        E_pen = self.penetration_energy_improved()*1000
        E_trans = self.punish_transform()*0
        # print(f"E_joint: {E_joint}, E_spen: {E_spen}, E_cont_1: {E_cont_1}, E_cont_2: {E_cont_2}, E_pen: {E_pen}, E_trans: {E_trans}")
        energy = E_joint + E_spen + E_cont_1 + E_cont_2 + E_cont_3 + E_cont_4 + E_pen + E_trans
        with torch.no_grad():
            energy_dict['E_joint'].append(E_joint)
            energy_dict['E_spen'].append(E_spen)
            energy_dict['E_cont_1'].append(E_cont_1)
            energy_dict['E_cont_2'].append(E_cont_2)
            energy_dict['E_cont_3'].append(E_cont_3)
            energy_dict['E_cont_4'].append(E_cont_4)
            energy_dict['E_pen'].append(E_pen)
            energy_dict['E_trans'].append(E_trans)
        self.energy = energy
        return energy

    def step(self, energy_dict = None):
        self.optimizer.zero_grad()
        self.q_current = torch.cat([self.test_translation*(1/self.translation_weight), 
                                    self.test_rotation*(1/self.rotation_weight), 
                                    self.test_joint*(1/self.joint_weight)], dim=1)
        self.handmodel.update_kinematics(q=self.q_current)
        energy = self.compute_energy(energy_dict) if self.global_step >= self.ik_step else self.compute_energy_init(energy_dict)
        energy.mean().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

    def get_opt_q(self):
        return self.q_current.detach()
