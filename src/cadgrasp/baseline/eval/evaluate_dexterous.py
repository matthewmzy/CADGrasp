import os
import sys

from isaacgym import gymapi, gymtorch
import yaml
import argparse
import numpy as np
import transforms3d
import torch
import xml.etree.ElementTree as ET
from pytorch3d.transforms import matrix_to_euler_angles
from pytorch3d import transforms as pttf
from typing import Union

from cadgrasp.baseline.utils.util import set_seed
from cadgrasp.baseline.utils.data_evaluator.data_evaluator import get_evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_list', type=str, nargs='*', 
        default=[
            'experiments/dex_ours/ckpt/ckpt_50000.pth', 
            ])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--robot_name', type=str, 
        default='leap_hand', choices=['leap_hand'])
    parser.add_argument('--scene_id', type=str, 
        default='scene_0100')
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--evaluator', type=str, default='SimulationEvaluator', 
        choices=['SimulationEvaluator'])
    parser.add_argument('--headless', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--split', type=str, default='')
    parser.add_argument('--dataset', type=str,
        default='graspnet', choices=['graspnet', 'acronym', 'combined'])
    parser.add_argument('--strategy', type=str,
        default='ours', choices=['ours', 'top10', 'graspness','logprob','random'])
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # load scene annotation
    if args.dataset == 'graspnet':
        scene_path = os.path.join('data/scenes', args.scene_id)
        extrinsics_path = os.path.join(scene_path, 'realsense/cam0_wrt_table.npy')
        extrinsics = np.load(extrinsics_path)
        annotation_path = os.path.join(scene_path, 'realsense/annotations/0000.xml')
        annotation = ET.parse(annotation_path)
    
        # parse scene annotation
        object_pose_dict = {}
        for obj in annotation.findall('obj'):
            object_code = str(int(obj.find('obj_id').text)).zfill(3)
            translation = np.array([float(x) for x in obj.find('pos_in_world').text.split()])
            rotation = np.array([float(x) for x in obj.find('ori_in_world').text.split()])
            rotation = transforms3d.quaternions.quat2mat(rotation)
            object_pose = np.eye(4)
            object_pose[:3, :3] = rotation
            object_pose[:3, 3] = translation
            object_pose = extrinsics @ object_pose
            object_pose_dict[object_code] = object_pose
        
        # load object surface points
        object_surface_points_dict = {}
        for object_code in object_pose_dict:
            object_surface_points_path = os.path.join('data/meshdata', 
                object_code, f'surface_points_1000.npy')
            object_surface_points = np.load(object_surface_points_path)
            object_pose = object_pose_dict[object_code]
            object_surface_points = object_surface_points @ object_pose[:3, :3].T + object_pose[:3, 3]
            object_surface_points_dict[object_code] = object_surface_points
    elif args.dataset == 'acronym':
        split = args.scene_id.split('_')[1]
        root = f'data/acronym_test_scenes/test_acronym_{split}'
        annotation_path = os.path.join(root,args.scene_id+'.npz')
        annotation = np.load(annotation_path,allow_pickle=True)['arr_0'][None][0]
        
        # parse scene annotation
        object_pose_dict = {}
        for obj in list(annotation.keys()):
            object_code = obj
            translation = annotation[object_code]['rest_pose_trans']
            translation[..., 2] -= 0.05
            rotation_quat = annotation[object_code]['rest_pose_quat']
            rotation_mat = pttf.quaternion_to_matrix(torch.tensor(rotation_quat[[3, 0, 1, 2]])).numpy()
            object_pose = np.eye(4)
            object_pose[:3, :3] = rotation_mat
            object_pose[:3, 3] = translation
            object_pose_dict[object_code] = object_pose
            
         # load object surface points
        object_surface_points_dict = {}
        for object_code in object_pose_dict:
            object_surface_points_path = os.path.join('data/acronym/meshes/pc', object_code, f'surface_points_1000.npy')
            object_surface_points = np.load(object_surface_points_path)
            object_pose = object_pose_dict[object_code]
            object_surface_points = object_surface_points @ object_pose[:3, :3].T + object_pose[:3, 3]
            object_surface_points_dict[object_code] = object_surface_points
    # create data evaluator
    evaluator_config_path = os.path.join(
        'configs/data_evaluator', args.robot_name, f'{args.evaluator}.yaml')
    evaluator_config = yaml.safe_load(open(evaluator_config_path, 'r'))
    evaluator_config['headless'] = args.headless
    evaluator_class = get_evaluator(args.evaluator)
    data_evaluator = evaluator_class(evaluator_config, args.device)
    
    # set environments
    data_evaluator.set_environments(object_pose_dict, object_surface_points_dict, args.batch_size + 1, dataset=args.dataset)
    
    # evaluate networks
    for ckpt_path in args.ckpt_path_list:
        # not overwrite
        if args.dataset == 'graspnet':
            save_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 
                f'results', args.scene_id, 'sim_success.npy')
        elif args.dataset == 'acronym':
            save_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'results_acronym', args.scene_id, 'sim_success.npy')
        if os.path.exists(save_path) and not args.overwrite:
            # print(f'{save_path} already exists')
            continue
        
        # load grasps
        if args.dataset == 'graspnet':
            load_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 
                'results', args.scene_id, f'grasps.npz')
            grasps = np.load(load_path)
        elif args.dataset == 'acronym':
            load_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'results_acronym', args.scene_id, 'grasps.npz')
            grasps = np.load(load_path)
        
        # evaluate grasps by batch
        successes = []
        for i in range(0, len(grasps['translation']), args.batch_size):
            end = min(i + args.batch_size, len(grasps['translation']))
            grasps_batch = { joint: grasps[joint][i:end] for joint in grasps }
            # pad the first grasp to avoid isaac gym bug
            grasps_batch = { joint: np.concatenate([grasps_batch[joint][:1], grasps_batch[joint]]) 
                for joint in grasps_batch }
            successes_batch = data_evaluator.evaluate_data(grasps_batch)
            successes.append(successes_batch[1:])
        
        # save results
        successes = np.concatenate(successes)
        print(np.mean(successes))
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        np.save(save_path, successes)

