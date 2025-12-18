import os
import sys
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from plotly import graph_objects as go
from ipdb import set_trace
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('.')

from IBSGrasp.scripts.scene import Scene
from IBSGrasp.utils.transforms import batch_transform_points, transform_points, pc_plotly, show
from PoseOptimize.utils.AdamOpt import AdamOptimizer
from PoseOptimize.utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from PoseOptimize.utils.HandModel import get_handmodel
from PoseOptimize.utils.visualize import save_energy_curve, visualize_traj_k3d
from PoseOptimize.utils.utils_math import interpolate_traj
from LASDiffusion.utils.visualize_ibs_vox import devoxelize
from LASDiffusion.generate import generate_based_on_scene_obs

import torch
import numpy as np
import copy
import yaml
import time

def IBSPolicy(batch_scene_pc_c, pred_ibs2cam, cam2world, args, with_traj=False, with_energy=False, with_ibs=False):
    """
    生成相机坐标系下的抓取位姿，通过IBS优化。

    Args:
        batch_scene_pc_c (torch.Tensor): 形状 (B, N, 3)，相机坐标系下的点云
        pred_ibs2cam (torch.Tensor): 形状 (B, 4, 4)，从相机坐标系到hand frame的变换
        cam2world: 仅用于可视化场景mesh时的transformation
        args: 参数命名空间，包含模型路径和优化参数

    Returns:
        tuple: (rotations, translations, qpos, valid_idx)
            - rotations (np.ndarray): 形状 (B, 3, 3)，相机坐标系下的旋转矩阵
            - translations (np.ndarray): 形状 (B, 3)，相机坐标系下的平移向量
            - qpos (np.ndarray): 形状 (B, n_joints)，关节位置
            - valid_idx (torch.Tensor): 有效的抓取索引
    """
    device = batch_scene_pc_c.device
    writer = SummaryWriter(args.logs_path)
    # 加载可视化需要用到的场景类
    if args.vis_ibs or args.vis_result_h or args.vis_result_c or args.vis_result_w:
        scene_cfg = DictConfig(yaml.safe_load(open('IBSGrasp/conf/scene.yaml','r')))
        scene_cfg.scene_id = args.scene_id
        scene_cfg.device = args.device
        scene_cfg.grasp_num = 0
        scene_cfg.load_gt_ibs = True
        scene_cfg.with_graspness = False
        scene = Scene(scene_cfg)
        world2ibs = torch.matmul(torch.inverse(pred_ibs2cam), torch.inverse(torch.from_numpy(cam2world).float().to(device)))
        world2ibs = world2ibs.detach().cpu().numpy()
        batch_scene_pc_h = batch_transform_points(batch_scene_pc_c,torch.inverse(pred_ibs2cam)).detach().cpu().numpy()

    # 预测IBS
    ibs_result = generate_based_on_scene_obs(
        model_path=args.las_model_path,
        steps=args.steps,
        hand_pose=pred_ibs2cam,
        scene_pc=batch_scene_pc_c
    )

    # 可视化IBS
    
    if args.vis_ibs:
        
        vis_cnt=0
        for ibs, scene_pc, s_trans in zip(ibs_result, batch_scene_pc_h, world2ibs):
            oc_pc, co_pc, th_pc = devoxelize(ibs)
            show([
                scene.get_plotly_mesh(transform=s_trans),
                pc_plotly(oc_pc, color='red'),
                pc_plotly(co_pc, color='yellow'),
                pc_plotly(th_pc, color='blue'),
                pc_plotly(scene_pc, color='grey'),
            ])
            vis_cnt+=1
            if vis_cnt==args.vis_ibs: break

    # 优化抓取位姿
    optimizer = AdamOptimizer(
        hand_name=args.hand_name,
        ibs_goal=ibs_result,
        writer = writer,
        max_iters=args.max_iters,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        decay_every=args.lr_decay_every,
        parallel_num=args.parallel_num,
        device=device
    )
    q_trajectory, energy_dict, valid_idx, valid_ibs = optimizer.run_adam(
        ibs_goal=ibs_result,
        running_name=f"IBSPolicy",
        cone_viz_num=args.cone_viz_num,
        cone_mu=args.cone_mu,
        filt_or_not=args.filt_or_not
    )

    # 获取最终位姿
    final_pose = q_trajectory[:, -1, :].detach().cpu().numpy()

    # 可视化抓取pose和IBS(手坐标系)
    if args.vis_result_h:
        vis_cnt = 0
        for i in range(len(valid_idx)):
            ibs = valid_ibs[i]
            scene_pc = batch_scene_pc_h[i]
            s_trans = world2ibs[i]
            hand_model = get_handmodel(args.hand_name, final_pose.shape[0], 'cpu', 1., use_collision=False)
            hand2ibs_25d = torch.from_numpy(final_pose).to(torch.float32)
            hand_plot = hand_model.get_plotly_data(hand2ibs_25d, i, color='red')
            oc_pc, co_pc, th_pc = devoxelize(ibs)
            show([
                scene.get_plotly_mesh(transform=s_trans),
                pc_plotly(oc_pc, color='red'),
                pc_plotly(co_pc, color='yellow'),
                pc_plotly(th_pc, color='blue'),
                pc_plotly(scene_pc, color='grey'),
            ] + hand_plot)
            vis_cnt+=1
            if vis_cnt==args.vis_result_h: break

    # 可视化抓取pose和IBS(camera space)
    if args.vis_result_c:
        vis_cnt = 0
        for i in range(len(valid_idx)):
            ibs = valid_ibs[i]
            scene_pc = batch_scene_pc_c[i].cpu().numpy() # camera space
            hand_model = get_handmodel(args.hand_name, final_pose.shape[0], 'cpu', 1., use_collision=False)
            hand2ibs_25d = torch.from_numpy(final_pose).to(torch.float32)
            hand2cam_25d = copy.deepcopy(hand2ibs_25d)
            temp_trans = hand2ibs_25d[:,:3]
            temp_rot = robust_compute_rotation_matrix_from_ortho6d(hand2ibs_25d[:,3:9])
            temp_transform = torch.eye(4).unsqueeze(0).repeat(temp_trans.shape[0],1,1).to(device=pred_ibs2cam.device)
            temp_transform[:,:3,3] = temp_trans
            temp_transform[:,:3,:3] = temp_rot
            temp_transform = pred_ibs2cam[valid_idx] @ temp_transform
            hand2cam_25d[:,:3] = temp_transform[:,:3,3]
            # from ipdb import set_trace; set_trace()
            hand2cam_25d[:,3:9] = temp_transform[:,:3,:3].permute(0,2,1)[:,:2].reshape(-1,6)
            hand_cam = hand_model.get_plotly_data(hand2cam_25d, i, color='red') # IBS space
            oc_pc, co_pc, th_pc = devoxelize(ibs)
            oc_pc = transform_points(oc_pc, pred_ibs2cam[i].cpu().numpy())
            co_pc = transform_points(co_pc, pred_ibs2cam[i].cpu().numpy())
            th_pc = transform_points(th_pc, pred_ibs2cam[i].cpu().numpy())
            show([
                scene.get_plotly_mesh(transform=pred_ibs2cam[i].cpu().numpy() @ world2ibs[i]),
                pc_plotly(oc_pc, color='red'),
                pc_plotly(co_pc, color='yellow'),
                pc_plotly(th_pc, color='blue'),
                pc_plotly(scene_pc, color='grey'),
            ] + hand_cam)
            vis_cnt+=1
            if vis_cnt==args.vis_result_c: break

    # 提取平移和旋转
    translations = final_pose[:, :3]
    rotations = robust_compute_rotation_matrix_from_ortho6d(torch.from_numpy(final_pose[:, 3:9])).cpu().numpy()
    # 提取关节位置
    qpos = final_pose[:, 9:]

    # 在hand frame下构建变换矩阵
    transforms = np.eye(4)[np.newaxis, :, :].repeat(len(final_pose), axis=0)
    transforms[:, :3, 3] = translations
    transforms[:, :3, :3] = rotations

    # 转换到相机坐标系
    pred_ibs2cam_np = pred_ibs2cam[valid_idx].cpu().numpy()
    camera_transforms = pred_ibs2cam_np @ transforms
    camera_translations = camera_transforms[:, :3, 3]
    camera_rotations = camera_transforms[:, :3, :3]
    
    # 在世界坐标系下可视化
    # 可视化抓取pose和IBS(world space)
    if args.vis_result_w:
        vis_cnt = 0
        for i in range(len(valid_idx)):
            ibs = valid_ibs[i]
            scene_pc_c = batch_scene_pc_c[i].cpu().numpy() # camera space
            scene_pc_w = transform_points(scene_pc_c, cam2world[valid_idx][i])
            hand_model = get_handmodel(args.hand_name, final_pose.shape[0], 'cpu', 1., use_collision=False)
            hand2ibs_25d = torch.from_numpy(final_pose).to(torch.float32)
            hand2world_25d = copy.deepcopy(hand2ibs_25d)
            temp_trans = hand2ibs_25d[:,:3]
            temp_rot = robust_compute_rotation_matrix_from_ortho6d(hand2ibs_25d[:,3:9])
            temp_transform = torch.eye(4).unsqueeze(0).repeat(temp_trans.shape[0],1,1).to(device=pred_ibs2cam.device)
            temp_transform[:,:3,3] = temp_trans
            temp_transform[:,:3,:3] = temp_rot
            temp_transform = torch.from_numpy(cam2world[valid_idx]).float().to(device=temp_transform.device) @ pred_ibs2cam[valid_idx] @ temp_transform
            hand2world_25d[:,:3] = temp_transform[:,:3,3]
            # from ipdb import set_trace; set_trace()
            hand2world_25d[:,3:9] = temp_transform[:,:3,:3].permute(0,2,1)[:,:2].reshape(-1,6)
            hand_world = hand_model.get_plotly_data(hand2world_25d, i, color='red') # IBS space
            oc_pc, co_pc, th_pc = devoxelize(ibs)
            pred_ibs2world = cam2world[valid_idx][i] @ pred_ibs2cam[valid_idx][i].cpu().numpy()
            oc_pc = transform_points(oc_pc, pred_ibs2world)
            co_pc = transform_points(co_pc, pred_ibs2world)
            th_pc = transform_points(th_pc, pred_ibs2world)
            show([
                scene.get_plotly_mesh(),
                pc_plotly(oc_pc, color='red'),
                pc_plotly(co_pc, color='yellow'),
                pc_plotly(th_pc, color='blue'),
                pc_plotly(scene_pc_w, color='grey'),
            ] + hand_world)
            vis_cnt+=1
            if vis_cnt==args.vis_result_w: break


    ret_list = [camera_rotations, camera_translations, qpos, valid_idx]
    if with_traj:
        ret_list.append(q_trajectory)
    if with_energy:
        ret_list.append(energy_dict)
    if with_ibs:
        ret_list.append(valid_ibs)
    return ret_list

@hydra.main(version_base="v1.2", config_path='../../conf', config_name='predict')
def main(cfg: DictConfig):
    args = OmegaConf.to_container(cfg, resolve=True)
    args = OmegaConf.create({k: v for d in args.values() for k, v in d.items()})
    writer = SummaryWriter(args.logs_path)
    
    ''' 加载数据 '''
    scene_cfg = DictConfig(yaml.safe_load(open('IBSGrasp/conf/scene.yaml', 'r')))
    scene_cfg.scene_id = args.scene_id
    scene_cfg.device = args.device
    scene_cfg.grasp_num = 0
    scene_cfg.load_gt_ibs = True
    scene_cfg.with_graspness = False
    scene = Scene(scene_cfg)

    batch_scene_pc_c, cam2world = scene.get_view_pc(args.view_id, downsample_pc_n=args.scene_pc_N)
    batch_scene_pc_c = batch_scene_pc_c.repeat(args.grasp_num, 1, 1)  # (B, N, 3) torch
    cam2world = cam2world[None].repeat(args.grasp_num, axis=0)       # (B, 4, 4) numpy
    data_perm = np.random.permutation(scene.ibs_data_num)[:args.grasp_num]
    pred_RT_frame_w = scene.w2h_trans[data_perm]
    pred_ibs2cam = np.linalg.inv(cam2world) @ np.linalg.inv(pred_RT_frame_w)  # (B, 4, 4) numpy

    ''' 转换为torch张量 '''
    pred_ibs2cam_torch = torch.from_numpy(pred_ibs2cam).float().to(args.device)
    batch_scene_pc_c = batch_scene_pc_c.to(args.device)

    rotations, translations, qpos, valid_idx, q_trajectory, energy_dict, ibs_result = IBSPolicy(
        batch_scene_pc_c=batch_scene_pc_c,
        pred_ibs2cam=pred_ibs2cam_torch,
        cam2world=cam2world,
        args=args,
        with_traj=True,
        with_energy=True,
        with_ibs=True
    )

    world_transforms = cam2world[valid_idx] @ np.concatenate(
        [np.concatenate([rotations, translations[:, :, None]], axis=2), np.ones((len(rotations), 1, 4))], axis=1
    )
    world_rotations = world_transforms[:, :3, :3]
    world_translations = world_transforms[:, :3, 3]

    # 保存轨迹和能量曲线
    if args.save_traj_num > 0:
        time_str = time.ctime().replace(' ','_').replace(':','_').replace('__','_')
        traj_save_root = os.path.join(args.traj_save_root, args.dataset, scene.scene_name, time_str)
        if not os.path.exists(traj_save_root):
            os.makedirs(traj_save_root)
        for i in range(min(args.save_traj_num, world_transforms.shape[0])):
            save_energy_curve(scene, energy_dict, i, traj_save_root)
            visualize_traj_k3d(scene, ibs_result[i], args.hand_name, q_trajectory, i, save_path=os.path.join(traj_save_root, f'traj_{i}.html'), world2ibs=world2ibs[i])


    ''' 保存结果（可选） '''
    if args.save_results:
        np.save(os.path.join(args.logs_path, 'rotations.npy'), world_rotations)
        np.save(os.path.join(args.logs_path, 'translations.npy'), world_translations)
        np.save(os.path.join(args.logs_path, 'qpos.npy'), qpos)

if __name__ == "__main__":
    main()