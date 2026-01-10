import os
import sys

import argparse
import multiprocessing
from rich.progress import track
import json

def predict_grasps(scene_id):
    
    # get worker id
    worker = multiprocessing.current_process()._identity[0]
    gpu = args.gpu_list[worker - 1]
    
    # command line
    command = ' '.join([
        f'CUDA_VISIBLE_DEVICES={gpu}', 
        'python src/eval/predict_dexterous.py',
        f'--ckpt_path {args.ckpt_path}',
        f'--device cuda:0',
        f'--urdf_path {args.urdf_path}',
        f'--meta_path {args.meta_path}',
        f'--camera {args.camera}',
        f'--scene_id {scene_id}',
        f'--grasp_num {args.grasp_num}',
        f'--seed {args.seed}',
        f'--overwrite {args.overwrite}',
        f'--stride {args.stride}',
        f'--scene_num {args.scene_num}',
        f'--dataset {args.dataset}',
        f'--all_scene_ids_acronym {args.all_scene_ids_acronym}',
        f'--strategy {args.strategy}',
        # ' > /dev/null 2>&1',
    ])
    
    # execute command
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, 
        default='experiments/v3.3_leap_diffvel/ckpt/ckpt_50000.pth')
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7]*2)
    parser.add_argument('--urdf_path', type=str, 
        default='robot_models/urdf/leap_hand.urdf')
    parser.add_argument('--meta_path', type=str, 
        default='robot_models/meta/leap_hand/meta.yaml')
    parser.add_argument('--camera', type=str, 
        default='realsense')
    parser.add_argument('--scene_id_start', type=int, default=100)
    parser.add_argument('--scene_id_end', type=int, default=190)
    parser.add_argument('--grasp_num', type=int, 
        default=1024)
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--scene_num', type=int, default=None)
    parser.add_argument('--dataset', type=str,
        default='acronym', choices=['graspnet', 'acronym'])
    parser.add_argument('--all_scene_ids_acronym', type=str, nargs='*',
        default=None)
    parser.add_argument('--strategy', type=str,
        default='ours', choices=['ours', 'top10', 'graspness','logprob','random'])
    args = parser.parse_args()
    
    # get scene id list
    if args.dataset == 'graspnet':
        if args.scene_id_start < 8500:
            if args.scene_num is None:
                args.scene_num = int((args.scene_id_end - args.scene_id_start) / len(args.gpu_list)) + 1
            scene_id_list = [f'scene_{str(i).zfill(4)}' for i in range(args.scene_id_start, args.scene_id_end, args.scene_num)]
        else:
            if args.scene_num is None:
                args.scene_num = int((args.scene_id_end - args.scene_id_start) / 5 / len(args.gpu_list)) + 1
            scene_id_list = [f'scene_{str(i).zfill(4)}' for i in range(args.scene_id_start, args.scene_id_end, args.scene_num*5)]
    elif args.dataset == 'acronym':
        scene_name_list = []
        scene_name_list += [f'scene_dense_{i}' for i in range(100)]
        scene_name_list += [f'scene_random_{i}' for i in range(90)]
        scene_name_list += [f'scene_loose_{i}' for i in range(30)]
        args.all_scene_ids_acronym = scene_name_list
        args.scene_num = int(len(args.all_scene_ids_acronym) / len(args.gpu_list)) + 1
        scene_id_list = range(0, len(scene_name_list), args.scene_num)
    
    # compose scenes in parallel
    # raise ValueError((len(scene_id_list)))
    with multiprocessing.Pool(len(args.gpu_list)) as pool:
        it = track(
            pool.imap_unordered(predict_grasps, scene_id_list), 
            total=len(scene_id_list), 
            description='predicting', 
        )
        list(it)
