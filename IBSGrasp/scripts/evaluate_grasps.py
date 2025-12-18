import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

from src.utils.data_evaluator.simulation_evaluator import SimulationEvaluator
import yaml
import numpy as np
import transforms3d

import xml.etree.ElementTree as ET
from pytorch3d.transforms import matrix_to_euler_angles
from pytorch3d import transforms as pttf
from typing import Union
from ipdb import set_trace
from termcolor import cprint
import hydra
from omegaconf import DictConfig

from src.utils.util import set_seed
# from src.utils.data_evaluator.data_evaluator import get_evaluator

import torch

@hydra.main(version_base="v1.2", config_path='../conf', config_name='eval')
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    save_path = os.path.join(args.save_path, args.scene_id, 'success_grasps.npz')
    
    if os.path.exists(save_path) and args.first_run:
        cprint(f'{save_path} already exists', 'red')
        return
    
    # load scene annotation
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
    
    # load grasps
    if args.first_run:
        scene_cfg = yaml.safe_load(open(args.scene_cfg_path, 'r'))
        scene_cfg['scene'] = args.scene_id
        from IBSGrasp.scripts.scene import Scene
        scene = Scene(DictConfig(scene_cfg))
        grasps = {'translation': [], 'rotation': []}
        grasp_points = []
        for obj_code, data_dict in scene.grasp_data.items():
            grasps['translation'].append(data_dict['trans'])
            grasps['rotation'].append(data_dict['rot'])
            grasp_points.append(data_dict['gp'])
            for joint_name, joint_tensor in data_dict['qpos'].items():
                if joint_name not in grasps.keys():
                    grasps[joint_name] = []
                grasps[joint_name].append(joint_tensor)
        grasps = {k:torch.concat(v, dim=0) for k,v in grasps.items()}
        grasp_points = torch.concat(grasp_points, dim=0)
    else:
        grasps = np.load(save_path)

    # create data evaluator
    evaluator_config_path = os.path.join(
        'configs/data_evaluator', args.robot_name, f'{args.evaluator}.yaml')
    evaluator_config = yaml.safe_load(open(evaluator_config_path, 'r'))
    evaluator_config['headless'] = args.headless
    data_evaluator = SimulationEvaluator(evaluator_config, device)
    
    # set environments
    data_evaluator.set_environments(object_pose_dict, 
                                    object_surface_points_dict, 
                                    args.batch_size + 1, 
                                    'graspnet')

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
    cprint(np.mean(successes), "green")
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    success_grasps = {k:v.detach().cpu().numpy()[successes] for k,v in grasps.items()}
    success_grasps['grasppoints'] = grasp_points.cpu().numpy()[successes]
    np.savez(save_path, success_grasps)

if __name__ == '__main__':
    main()