import os
import sys

import argparse
import numpy as np
import torch
from tqdm import trange

from cadgrasp.baseline.utils.robot_model import RobotModel
from cadgrasp.baseline.utils.util import set_seed
from cadgrasp.baseline.utils.config import ckpt_to_config
from cadgrasp.baseline.utils.dataset import get_sparse_tensor
from cadgrasp.baseline.network.model import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, 
        default='experiments/v3.3_leap_diffvel_noacronym/ckpt/ckpt_50000.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--urdf_path', type=str, 
        default='robot_models/urdf/leap_hand_simplified.urdf')
    parser.add_argument('--meta_path', type=str, 
        default='robot_models/meta/leap_hand/meta.yaml')
    parser.add_argument('--camera', type=str, 
        default='realsense')
    parser.add_argument('--scene_id', type=str, 
        default='acronym_novel_00004')
    parser.add_argument('--all_scene_ids_acronym', type=str, nargs='*',
        default=None)
    parser.add_argument('--grasp_num', type=int, 
        default=1024)
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--scene_num', type=int, default=10)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--dataset', type=str,
        default='acronym', choices=['graspnet', 'acronym'])
    parser.add_argument('--strategy', type=str,
        default='random', choices=['ours', 'top10', 'graspness','logprob','random'])
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # not overwrite
   
    # robot model
    robot_model = RobotModel(args.urdf_path, args.meta_path)
    
    # load model
    config = ckpt_to_config(args.ckpt_path)
    model = get_model(config.model)
    model.config.voxel_size = config.data.voxel_size
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    if args.dataset == 'graspnet':
        start_idx = int(args.scene_id.split('_')[-1])
    elif args.dataset == 'acronym':
        start_idx = int(args.scene_id)
    for idx in trange(args.scene_num):
        if args.dataset == 'graspnet':
            if start_idx < 190:
                if start_idx + idx >= 190:
                    break
                args.scene_id = f'scene_{start_idx + idx:04d}'
                print(f'Processing {args.scene_id}')
            elif start_idx < 380:
                if start_idx + idx >= 380:
                    break
                args.scene_id = f'scene_{start_idx + idx:04d}'
                print(f'Processing {args.scene_id}')
            elif start_idx > 8500:
                if start_idx + idx*5 >= 9900:
                    break
                args.scene_id = f'scene_{start_idx + idx*5:04d}'
                print(f'Processing {args.scene_id}')
            save_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 
                'results', args.scene_id, f'grasps.npz')
            if os.path.exists(save_path) and not args.overwrite:
                continue
            
            # load network input
            load_path = os.path.join('data/scenes', args.scene_id, args.camera, 'network_input.npz')
        elif args.dataset == 'acronym':
            all_scene_ids = args.all_scene_ids_acronym
            if start_idx + idx >= len(all_scene_ids):
                break
            args.scene_id = all_scene_ids[start_idx + idx].strip(',').strip('[').strip(']')
            print(f'Processing {args.scene_id}')
            split = args.scene_id.split('_')[1]
            load_path = os.path.join(f'data/acronym_test_scenes/network_input_{split}', args.scene_id, args.camera, 'network_input.npz')
            
            save_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results_acronym', args.scene_id, f'grasps.npz')
            if os.path.exists(save_path) and not args.overwrite:
                continue
        
        try:
            network_input = dict(np.load(load_path))
            pc_all = torch.tensor(network_input['pc'], dtype=torch.float)
            seg_all = torch.tensor(network_input['seg'], dtype=torch.long)
            edge_all = torch.tensor(network_input['edge'], dtype=torch.long)
            extrinsics_all = network_input['extrinsics']
            
            # voxelize
            
            # predict grasps
            with torch.no_grad():
                rotations, translations, qposs, scores, obj_indicess, graspnesses, log_probs = [], [], [], [], [], [], []
                stride = args.stride
                for i in range(0, 256, stride):
                    data_part = get_sparse_tensor(pc_all[i:i+stride], config.data.voxel_size)
                    data_part['seg'] = seg_all[i:i+stride]
                    data_part = {k: v.to(device) for k, v in data_part.items()}
                    edge_part = edge_all[i:i+stride]
                    rotation, translation, qpos, score, obj_indices, graspness, log_prob = \
                        (t.cpu() for t in model.sample(data_part, args.grasp_num, graspness_scale=5, allow_fail=True, cate=False, edge=edge_part.to(device), with_score_parts=True))
                    rotations.append(rotation)
                    translations.append(translation)
                    qposs.append(qpos)
                    scores.append(score)
                    obj_indicess.append(obj_indices)
                    graspnesses.append(graspness)
                    log_probs.append(log_prob)
                rotation, translation, qpos, score, obj_indices, graspness, log_prob = map(lambda x: torch.cat(x, dim=0), 
                    [rotations, translations, qposs, scores, obj_indicess, graspnesses, log_probs])
            
            # select best
            if True:  # if args.maxrange == 'top1':
                if args.strategy == 'ours':
                    best_indices = score.argmax(dim=1)
                elif args.strategy == 'top10':
                    best_indices = torch.topk(score,10,dim=1)[1]
                    best_indices_list = []
                    for i in range(best_indices.shape[0]):
                        best_indice = best_indices[i]
                        rand = torch.randperm(best_indices.shape[-1])[0][None]
                        best_indice = best_indice[...,rand]
                        best_indices_list.append(best_indice)
                    best_indices = torch.cat(best_indices_list)
                    
                elif args.strategy == 'graspness':
                    best_indices = graspness.argmax(dim=1)
                elif args.strategy == 'logprob':
                    best_indices = log_prob.argmax(dim=1)
                elif args.strategy == 'random':
                    best_indices = torch.randint(low=0,high=score.shape[1],size=(256,))
            
            arange = torch.arange(len(best_indices))
            sel_rotations = rotation[arange, best_indices].cpu().numpy()
            sel_translations = translation[arange, best_indices].cpu().numpy()
            sel_qposs = qpos[arange, best_indices].cpu().numpy()
            # transform to world frame
            sel_rotations = extrinsics_all[:, :3, :3] @ sel_rotations
            sel_translations = (extrinsics_all[:, :3, :3] @ sel_translations[:, :, None] + \
                extrinsics_all[:, :3, 3:])[:, :, 0]
            
            # save grasps
            grasps = {}
            grasps['rotation'] = sel_rotations
            grasps['translation'] = sel_translations
            sel_qposs_dict = { joint: sel_qposs[:, i] for i, joint in enumerate(robot_model.joint_names) }
            grasps.update(sel_qposs_dict)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **grasps)
        except Exception as e:
            print("skipped")
            print(e)
            pass
